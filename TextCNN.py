import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Convolutional Neural Networks for Sentence Classification
http://www.aclweb.org/anthology/D14-1181
"""


class TextCNN(nn.Module):
    def __init__(self, args, word_vec, n_classes):
        super(TextCNN, self).__init__()
        # V = args.embed_num
        # D = args.embed_dim
        # C = args.class_num
        # Ci = 1
        # Co = args.kernel_num
        # Ks = args.kernel_sizes
        V = word_vec.shape[0]
        D = word_vec.shape[1]
        C = n_classes
        Ci = 1
        self.Co = 1000
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(V, D)
        if args.pretrained_word_embed:
            self.embed.weight = nn.Parameter(torch.from_numpy(word_vec).float())
            self.embed.weight.requires_grad = args.update_word_embed
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, self.Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.)
        self.fc1 = nn.Linear(len(Ks) * self.Co, C)

    def forward(self, x, fc=False):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        if fc:
            x = self.fc1(x)  # (N, C)
        return x
