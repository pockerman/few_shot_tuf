import torch
import torch.nn as nn


def linear(in_features: int, out_features: int, bias: bool=True) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features=in_features,
                                   out_features=out_features, bias=bias),
                         nn.ReLU())


def linear_with_softmax(in_features: int, out_features: int, bias: bool=True) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features=in_features,
                                   out_features=out_features, bias=bias),
                         nn.ReLU(),
                         nn.Softmax())


def convolution_with_linear_softmax(in_channels: int, out_channels: int, kernel_size: int,
                                    in_features: int, out_features: int,
                                    bias: bool=True) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                         nn.Linear(in_features=in_features,
                                   out_features=out_features, bias=bias),
                         nn.ReLU(), nn.Softmax())


def conv_block_(in_channels: int, out_channels: int, padding: int) -> nn.Sequential:

    bn = nn.BatchNorm2d(out_channels)

    # for pytorch 1.2 or later
    nn.init.uniform_(bn.weight)
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                         bn, nn.ReLU(),
                         nn.MaxPool2d(2))


def conv_block(in_channels: int, out_channels: int, padding: int):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                         nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))
