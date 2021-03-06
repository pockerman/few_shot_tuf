from pathlib import Path

import csv
import numpy as np
import torch
import torch.utils.data as data

"""
def get_default_class_labels():
    return {"Duplicate": 0, "Full-Delete": 1,
            "Normal-1": 2, "Single-Delete": 3,
            "TUF": 4, "Normal-2": 5}
"""


class TUFDataset(data.Dataset):
    """
    Class representing TUF dataset
    """

    def __init__(self, filename: Path, dataset_type: str, classes,
                 do_load=True, device: str='cpu', transform: object=None) -> None:
        super(TUFDataset, self).__init__()
        self._filename = filename
        self._dataset_type = dataset_type
        self._transform = transform
        self._X = None
        self._y = None
        self._classes = classes
        self._n_classes = 0
        self._device = device

        if do_load:
            self.load()

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def unique_classes(self):
        return np.unique(self._y)

    @property
    def labels(self):
        return self._y

    def load(self):
        """
        Load the dataset
        """

        with open(self._filename, 'r', newline='\n') as fh:
            csv_reader = csv.reader(fh, delimiter=',')

            X_tmp = []
            y_tmp = []
            for row in csv_reader:

                if len(row) == 0:
                    break

                if row[0] == '#':
                    self._n_classes = int(row[1].split(':')[-1])
                else:
                    # at the end is the label
                    label = int(row[-1])
                    y_tmp.append(int(label))
                    X_tmp.append([float(row[0]), float(row[1])])

            if len(X_tmp) == 0 or len(y_tmp) == 0:
                raise ValueError("Empty dataset or empty labels")

            self._X = torch.tensor(data=np.array(X_tmp), device=self._device).float()
            self._y = torch.tensor(data=np.array(y_tmp), device=self._device).float()

    def __getitem__(self, idx):
        """
        Returns the idx tuple: (example_point, example_label)
        """

        try:
            x = self._X[idx]

            if self._transform:
                x = self._transform(x)
            return x, self._y[idx]
        except IndexError as e:
            print(self._dataset_type)
            print(idx)
            print(self._X.shape)
            print(len(self))
            raise e

    def __len__(self):
        return len(self._X)


