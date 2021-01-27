"""
PyTorch based implementation of prototypical
network for TUF and DNA labeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetTUF(nn.Module):
    """
    Class defining Prototypical network
    """

    def __init__(self):
        super(PrototypicalNetTUF, self).__init__()

    def forward(self, x):
        """
        Forward pass on the data
        """
        pass