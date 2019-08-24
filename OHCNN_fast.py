import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
http://www.anthology.aclweb.org/N/N15/N15-1011.pdf
use equivalent embeddings for faster speed (2~3x)
"""


class OHCNN_fast(nn.Module):

    def __init__(self, unk_idx, n_classes, vocab_size):
        super(OHCNN_fast, self).__init__()
        # D = 30001
        print(f'vocab_size:{vocab_size}')
        D = vocab_size
        C = n_classes
        Co = 1000
        self.Co = Co
        self.n_pool = 10
        self.embed = nn.Embedding(D, Co)
        self.bias = nn.Parameter(torch.Tensor(1, Co, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(Co * self.n_pool, C)
        self.unk_idx = unk_idx
        # init as in cnn
        stdv = 1. / math.sqrt(D)
        self.embed.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, fc=False):
        # (32, 256, 20)
        sent_len = x.size(1)
        x = x.view(x.size(0), -1)
        x_embed = self.embed(x)  # (N, W * D, Co)
        # deal with unk in the region
        x = (x != self.unk_idx).float().unsqueeze(-1) * x_embed
        x = x.view(x.size(0), sent_len, -1, self.Co)  # (N, W, D, Co)
        x = F.relu(x.sum(2).permute(0, 2, 1) + self.bias)  # (N, Co, W)
        x = F.avg_pool1d(x, int(x.size(2) / self.n_pool)).view(-1, self.n_pool * self.Co)  # (N, n_pool * Co)
        x = self.dropout(x)
        # response norm
        x /= (1 + x.pow(2).sum(1)).sqrt().view(-1, 1)
        if fc:
            x = self.fc1(x)  # (N, C)
        return x
