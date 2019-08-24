import torch.nn as nn
import torch.nn.functional as F

"""
Linear model used for feature input (instead of text)
"""


class Linear_Model(nn.Module):
    def __init__(self, args, input_dim, n_classes):
        super(Linear_Model, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(input_dim, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, n_classes)

    def forward(self, x, fc=False):
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if fc:
            x = self.fc2(x)
        return x
