import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from IPython import embed

"""
Hierarchical Multi-Label Classification Networks
http://proceedings.mlr.press/v80/wehrmann18a.html
"""


class HMCN(nn.Module):
    def __init__(self, base_model, args, neuron_each_local_l2, total_class, in_dim):
        super(HMCN, self).__init__()

        neuron_each_layer = [384] * len(neuron_each_local_l2)
        neuron_each_local_l1 = [384] * len(neuron_each_local_l2)
        self.beta = 0.5

        self.args = args
        self.base_model = base_model

        self.layer_num = len(neuron_each_layer)
        self.linear_layers = nn.ModuleList([])
        self.local_linear_l1 = nn.ModuleList([])
        self.local_linear_l2 = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([])
        self.batchnorms_local_1 = nn.ModuleList([])
        for idx, neuron_number in enumerate(neuron_each_layer):
            if idx == 0:
                self.linear_layers.append(nn.Linear(in_dim, neuron_number))
            else:
                self.linear_layers.append(
                    nn.Linear(neuron_each_layer[idx - 1] + in_dim, neuron_number))
            self.batchnorms.append(nn.BatchNorm1d(neuron_number))

        for idx, neuron_number in enumerate(neuron_each_local_l1):
            self.local_linear_l1.append(
                nn.Linear(neuron_each_layer[idx], neuron_each_local_l1[idx]))
            self.batchnorms_local_1.append(
                nn.BatchNorm1d(neuron_each_local_l1[idx]))
        for idx, neuron_number in enumerate(neuron_each_local_l2):
            self.local_linear_l2.append(
                nn.Linear(neuron_each_local_l1[idx], neuron_each_local_l2[idx]))

        self.final_linear_layer = nn.Linear(
            neuron_each_layer[-1] + in_dim, total_class)

    def forward(self, x):
        x = self.base_model(x, False)
        local_outputs = []
        output = x
        for layer_idx, layer in enumerate(self.linear_layers):
            if layer_idx == 0:
                output = layer(output)
                output = F.relu(output)
            else:
                output = layer(torch.cat([output, x], dim=1))
                output = F.relu(output)

            local_output = self.local_linear_l1[layer_idx](output)
            local_output = F.relu(local_output)
            local_output = self.local_linear_l2[layer_idx](local_output)
            local_outputs.append(local_output)

        global_outputs = F.sigmoid(
            self.final_linear_layer(torch.cat([output, x], dim=1)))
        local_outputs = F.sigmoid(torch.cat(local_outputs, dim=1))

        output = self.beta * global_outputs + (1 - self.beta) * local_outputs

        return output
