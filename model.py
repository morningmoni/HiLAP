import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, args, n_class, base_model, in_dim):
        super(Policy, self).__init__()
        self.args = args
        self.L = 0.02
        self.baseline_reward = 0
        self.entropy_l = []
        self.beta = args.beta
        self.beta_decay_rate = .9
        self.n_update = 0
        self.class_embed = nn.Embedding(n_class, args.class_embed_size)
        self.class_embed_bias = nn.Embedding(n_class, 1)

        stdv = 1. / np.sqrt(self.class_embed.weight.size(1))
        self.class_embed.weight.data.uniform_(-stdv, stdv)
        self.class_embed_bias.weight.data.uniform_(-stdv, stdv)

        self.saved_log_probs = []
        self.rewards = []
        self.rewards_greedy = []
        self.doc_vec = None
        self.base_model = base_model

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.sl_loss = 0

        if self.args.use_history:
            self.state_hist = None
            self.output_hist = None
            self.hist_gru = nn.GRU(args.class_embed_size, args.class_embed_size, bidirectional=True)
        if self.args.use_cur_class_embed:
            in_dim += self.args.class_embed_size
        if self.args.use_history:
            in_dim += args.hist_embed_size * 2
        if self.args.use_l2:
            self.l1 = nn.Linear(in_dim, args.l1_size)
            self.l2 = nn.Linear(args.l1_size, args.class_embed_size)
        elif self.args.use_l1:
            self.l1 = nn.Linear(in_dim, args.class_embed_size)

    def update_baseline(self, target):
        # a moving average baseline, not used anymore
        self.baseline_reward = self.L * target + (1 - self.L) * self.baseline_reward

    def finish_episode(self):
        self.sl_loss = 0
        self.n_update += 1
        if self.n_update % self.args.update_beta_every == 0:
            self.beta *= self.beta_decay_rate
        self.entropy_l = []
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.rewards_greedy[:]

    def forward(self, cur_class_batch, next_classes_batch):
        cur_class_embed = self.class_embed(cur_class_batch)  # (batch, 50)
        next_classes_embed = self.class_embed(next_classes_batch)  # (batch, max_choices, 50)
        nb = self.class_embed_bias(next_classes_batch).squeeze(-1)
        states_embed = self.doc_vec
        if self.args.use_cur_class_embed:
            states_embed = torch.cat((states_embed, cur_class_embed), 1)
        if self.args.use_history:
            states_embed = torch.cat((states_embed, self.output_hist.squeeze()), 1)
        if not self.args.use_l1:
            return torch.bmm(next_classes_embed, states_embed.unsqueeze(-1)).squeeze(-1) + nb
        if self.args.use_l2:
            h1 = F.relu(self.l1(states_embed))
            h2 = F.relu(self.l2(h1))
        else:
            h2 = F.relu(self.l1(states_embed))
        h2 = h2.unsqueeze(-1)  # (batch, 50, 1)
        probs = torch.bmm(next_classes_embed, h2).squeeze(-1) + nb
        if self.args.use_history:
            self.output_hist, self.state_hist = self.hist_gru(cur_class_embed.unsqueeze(0), self.state_hist)
        return probs

    def duplicate_doc_vec(self, indices):
        assert self.doc_vec is not None
        assert len(indices) > 0
        self.doc_vec = self.doc_vec[indices]

    def duplicate_reward(self, indices):
        assert len(indices) > 0
        self.saved_log_probs[-1] = [[probs[i] for i in indices] for probs in self.saved_log_probs[-1]]
        self.rewards[-1] = [[R[i] for i in indices] for R in self.rewards[-1]]

    def generate_doc_vec(self, mini_batch):
        self.doc_vec = self.base_model(mini_batch)

    def generate_logits(self, mini_batch, cur_class_batch, next_classes_batch):
        if self.doc_vec is None:
            self.generate_doc_vec(mini_batch)
        if self.args.gpu:
            cur_class_batch = Variable(torch.from_numpy(cur_class_batch)).cuda()
            next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda()
        else:
            cur_class_batch = Variable(torch.from_numpy(cur_class_batch))
            next_classes_batch = Variable(torch.from_numpy(next_classes_batch))
        logits = self(cur_class_batch, next_classes_batch)
        # mask padding relations
        logits = (next_classes_batch == 0).float() * -99999 + (next_classes_batch != 0).float() * logits
        return logits

    def step_sl(self, mini_batch, cur_class_batch, next_classes_batch, next_classes_batch_true, sigmoid=True):
        logits = self.generate_logits(mini_batch, cur_class_batch, next_classes_batch)
        if not sigmoid:
            return logits
        if next_classes_batch_true is not None:
            if self.args.gpu:
                y_true = Variable(torch.from_numpy(next_classes_batch_true)).cuda().float()
            else:
                y_true = Variable(torch.from_numpy(next_classes_batch_true)).float()
            self.sl_loss += self.criterion(logits, y_true)
        return F.sigmoid(logits)

    def step(self, mini_batch, cur_class_batch, next_classes_batch, test=False, flat_probs=None):
        logits = self.generate_logits(mini_batch, cur_class_batch, next_classes_batch)
        if self.args.softmax:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = F.sigmoid(logits)
        if not test:
            # + epsilon to avoid log(0)
            self.entropy_l.append(torch.mean(torch.log(probs + 1e-32) * probs))
        next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda()
        probs = probs + (next_classes_batch != 0).float() * 1e-16
        m = Categorical(probs)
        if test or self.args.sample_mode == 'choose_max':
            action = torch.max(probs, 1)[1]
        elif self.args.sample_mode == 'random':
            if random.random() < 1.2:
                if self.args.gpu:
                    action = Variable(torch.zeros(probs.size()[0]).long().random_(0, probs.size()[1])).cuda()
                else:
                    action = Variable(torch.zeros(probs.size()[0]).long().random_(0, probs.size()[1]))
            else:
                action = m.sample()
        else:
            action = m.sample()
        return action, m

    # not used anymore
    def init_hist(self, tokens_size):
        if self.args.gpu:
            self.state_hist = self.init_hidden(tokens_size, self.args.hist_embed_size).cuda()
            first_input = Variable(
                torch.from_numpy(np.zeros((1, tokens_size, self.args.class_embed_size)))).cuda().float()
        else:
            self.state_hist = self.init_hidden(tokens_size, self.args.hist_embed_size)
            first_input = Variable(
                torch.from_numpy((np.zeros(1, tokens_size, self.args.class_embed_size)))).float()
        self.output_hist, self.state_hist = self.hist_gru(first_input, self.state_hist)
