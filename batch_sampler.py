import torch
import numpy as np


class BatchSampler(object):
    '''
    Initialize the BatchSampler object
    Args:
    - labels: an iterable containing all the labels for the current dataset
            samples indexes will be infered from this iterable.
    - classes_per_it: number of random classes for each iteration
    - num_samples: number of samples for each iteration for each class (support + query)
    - iterations: number of iterations (episodes) per epoch
    '''

    def __init__(self, labels, classes_per_it: int,
                 num_samples: int, iterations: int, mode: str) -> None:

        self._labels = labels
        self._classes_per_it = classes_per_it
        self._sample_per_class = num_samples
        self._iterations = iterations
        self._mode = mode

        self._classes, self._counts = np.unique(self._labels, return_counts=True)
        self._classes = torch.LongTensor(self._classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self._labels))
        self._indexes = np.empty((len(self._classes), max(self._counts)), dtype=int) * np.nan
        self._indexes = torch.Tensor(self._indexes)
        self._numel_per_class = torch.zeros_like(self._classes)

        for idx, label in enumerate(self._labels):
            label_idx = np.argwhere(self._classes == label).item()
            self._indexes[label_idx, np.where(np.isnan(self._indexes[label_idx]))[0][0]] = idx
            self._numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self._sample_per_class
        cpi = self._classes_per_it

        for it in range(self._iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self._classes))[:cpi]
            for i, c in enumerate(self._classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self._classes)).long()[self._classes == c].item()
                sample_idxs = torch.randperm(self._numel_per_class[label_idx])[:spc]
                batch[s] = self._indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self._iterations