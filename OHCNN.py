import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
http://www.anthology.aclweb.org/N/N15/N15-1011.pdf
"""


class OHCNN(nn.Module):

    def __init__(self, args, n_classes):
        super(OHCNN, self).__init__()
        self.args = args
        D = 30000
        C = n_classes
        Ci = 1
        Co = 1000
        self.Co = Co
        self.n_pool = 10
        if args.mode == 'ohcnn-seq':
            Ks = [3]
            self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K - 1, 0)) for K in Ks])
        else:
            Ks = [1]
            self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), stride=1) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        # self.lrn = nn.LocalResponseNorm(2)
        self.fc1 = nn.Linear(len(Ks) * Co * self.n_pool, C)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # [(N, Co * n_pool), ...]*len(Ks)
        x = [F.avg_pool1d(i, int(i.size(2) / self.n_pool)).view(-1, self.n_pool * self.Co) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # response norm
        x /= (1 + x.pow(2).sum(1)).sqrt().view(-1, 1)
        logit = self.fc1(x)  # (N, C)
        return logit
