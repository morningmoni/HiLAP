import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Hierarchical Attention Networks for Document Classification
https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
"""


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if nonlinearity == 'tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class HAN(nn.Module):
    def __init__(self, args, n_classes, word_vec=None, num_tokens=None, embed_size=None):
        super(HAN, self).__init__()
        self.args = args
        if word_vec is None:
            assert num_tokens is not None and embed_size is not None
            self.num_tokens = num_tokens
            self.embed_size = embed_size
        else:
            self.num_tokens = word_vec.shape[0]
            self.embed_size = word_vec.shape[1]
        self.word_gru_hidden = args.word_gru_hidden_size
        self.lookup = nn.Embedding(self.num_tokens, self.embed_size)
        if args.pretrained_word_embed:
            self.lookup.weight = nn.Parameter(torch.from_numpy(word_vec).float())
            self.lookup.weight.requires_grad = args.update_word_embed
        self.word_gru = nn.GRU(self.embed_size, self.word_gru_hidden, bidirectional=True)
        self.weight_W_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 2 * self.word_gru_hidden))
        self.bias_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))
        nn.init.uniform(self.weight_W_word, -0.1, 0.1)
        nn.init.uniform(self.bias_word, -0.1, 0.1)
        nn.init.uniform(self.weight_proj_word, -0.1, 0.1)
        # sentence level
        self.sent_gru_hidden = args.sent_gru_hidden_size
        self.word_gru_hidden = args.word_gru_hidden_size
        self.sent_gru = nn.GRU(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
        self.weight_W_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 2 * self.sent_gru_hidden))
        self.bias_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        C = n_classes
        self.fc1 = nn.Linear(2 * self.sent_gru_hidden, C)
        nn.init.uniform(self.bias_sent, -0.1, 0.1)
        nn.init.uniform(self.weight_W_sent, -0.1, 0.1)
        nn.init.uniform(self.weight_proj_sent, -0.1, 0.1)

    def forward(self, mini_batch, fc=False):
        max_sents, batch_size, max_tokens = mini_batch.size()
        word_attn_vectors = None
        state_word = self.init_hidden(mini_batch.size()[1])
        for i in range(max_sents):
            embed = mini_batch[i, :, :].transpose(0, 1)
            embedded = self.lookup(embed)
            output_word, state_word = self.word_gru(embedded, state_word)
            word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
            # logger.debug(word_squish.size()) torch.Size([20, 2, 200])
            word_attn = batch_matmul(word_squish, self.weight_proj_word)
            # logger.debug(word_attn.size()) torch.Size([20, 2])
            word_attn_norm = F.softmax(word_attn.transpose(1, 0), dim=-1)
            word_attn_vector = attention_mul(output_word, word_attn_norm.transpose(1, 0))
            if word_attn_vectors is None:
                word_attn_vectors = word_attn_vector
            else:
                word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
        # logger.debug(word_attn_vectors.size()) torch.Size([1, 2, 200])
        state_sent = self.init_hidden(mini_batch.size()[1])
        output_sent, state_sent = self.sent_gru(word_attn_vectors, state_sent)
        # logger.debug(output_sent.size()) torch.Size([8, 2, 200])
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        # logger.debug(sent_squish.size()) torch.Size([8, 2, 200])
        if len(sent_squish.size()) == 2:
            sent_squish = sent_squish.unsqueeze(0)
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        if len(sent_attn.size()) == 1:
            sent_attn = sent_attn.unsqueeze(0)
        # logger.debug(sent_attn.size())  torch.Size([8, 2])
        sent_attn_norm = F.softmax(sent_attn.transpose(1, 0), dim=-1)
        # logger.debug(sent_attn_norm.size()) torch.Size([2, 8])
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1, 0))
        # logger.debug(sent_attn_vectors.size()) torch.Size([1, 2, 200])
        x = sent_attn_vectors.squeeze(0)
        if fc:
            x = self.fc1(x)
        return x

    def init_hidden(self, batch_size, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.sent_gru_hidden
        if self.args.gpu:
            return Variable(torch.zeros(2, batch_size, hidden_dim)).cuda()
        return Variable(torch.zeros(2, batch_size, hidden_dim))
