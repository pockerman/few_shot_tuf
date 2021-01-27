"""
PyTorch based implementation of prototypical
network for TUF and DNA labeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         bn, nn.ReLU(), nn.MaxPool2d(2))


class PrototypicalNetTUF(nn.Module):
    """
    Class defining Prototypical network
    """

    @staticmethod
    def build_network(options):

        device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
        return PrototypicalNetTUF().to(device=device)

    def __init__(self):
        super(PrototypicalNetTUF, self).__init__()

    def forward(self, x):
        """
        Forward pass on the data
        """
        pass