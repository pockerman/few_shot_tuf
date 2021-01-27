import numpy as np
import torch


def init_seed(manual_seed):
    """
    Disable cudnn to maximize reproducibility
    """

    torch.cuda.cudnn_enabled = False
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)


def init_dataloader(dataset, sampler):
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader