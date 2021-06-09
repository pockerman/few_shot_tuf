"""
PyTorch based implementation of prototypical
network for TUF and DNA labeling. Implementation
is taken from
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from distances import euclidean_dist


def prototypical_loss(input, target, n_support):
    """
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target.
    It then computes the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing barycentres, for each one of the current classes
    """

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val


class PrototypicalLoss(nn.Module):
    """
    Loss class deriving from Module for the
    prototypical loss function defined below
    """

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


class ProtoNetTUF(nn.Module):
    """
    Class defining Prototypical network
    """

    @staticmethod
    def build_network(encoder: nn.Sequential, options: dict) -> object:
        return ProtoNetTUF(encoder=encoder).to(device=options["device"])

    def __init__(self, encoder: nn.Sequential = None) -> None:
        super(ProtoNetTUF, self).__init__()

        self._encoder = encoder
        
    def loss_fn(self, input, target, n_support):
        """
        Compute loss function
        """
        return prototypical_loss(input=input, target=target, n_support=n_support)

    def forward(self, x):
        """
        Forward pass on the data
        """
        x = self._encoder(x)
        return x.view(x.size(0), -1)

    def predict(self, x):
        self.train(mode=False)

        result = self.forward(x=x)
        return np.argmax(result.detach().numpy(), axis=1)
