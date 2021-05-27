import unittest
import pytest

from pathlib import Path
import torch

import torch.optim as optim
import torch.utils.data as data

from train_engine import TrainEngine
from prototypical_net_tuf import ProtoNetTUF
from tuf_dataset import TUFDataset
from batch_sampler import BatchSampler
from encoders import linear
from utils import init_seed


class TestTrainEngine(unittest.TestCase):

    def test_construction(self):

        try:
            train_engine = TrainEngine()
        except:
            self.fail("TrainEngine construction failed")

    def test_train_1(self):

        with pytest.raises(AssertionError) as error:
            train_engine = TrainEngine(model=None)
            train_engine.train(options=None)

        self.assertEqual(str(error.value), "Model has not been specified")

    def test_train_2(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF(encoder=None)
            train_engine = TrainEngine(model=proto_net)
            train_engine.train(options=None)

        self.assertEqual(str(error.value), "Training options not specified")

    def test_train_3(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF(encoder=None)
            train_engine = TrainEngine(model=proto_net)

            options = {"lr_scheduler": None}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Learning scheduler has not been specified")

    def test_train_4(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF(encoder=linear(in_features=2, out_features=1))
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)

            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01, step_size=0.5, verbose=True)

            options = {"optimizer": None, "lr_scheduler": lr_scheduler}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Optimizer has not been specified")

    def test_train_5(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF(encoder=linear(in_features=2, out_features=1))
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)
            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01, step_size=0.5,
                                                           verbose=True)

            options = {"optimizer": optimizer, "lr_scheduler": lr_scheduler,
                       "max_epochs": 0}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Invalid number of max epochs")

    def test_trainer_6(self):
        with pytest.raises(AssertionError) as error:
            proto_net = ProtoNetTUF(encoder=linear(in_features=2, out_features=1))
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)
            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01,
                                                           step_size=0.5,
                                                           verbose=True)

            options = {"optimizer": optimizer, "lr_scheduler": lr_scheduler,
                       "max_epochs": 2, "device": "cpu", "iterations": 0,
                       "sample_loader": None}
            train_engine.train(options=options)
        self.assertEqual(str(error.value), "Invalid number of iterations per epoch")

    def test_trainer_7(self):
        with pytest.raises(AssertionError) as error:
            proto_net = ProtoNetTUF(encoder=linear(in_features=2, out_features=3))
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)
            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01,
                                                           step_size=0.5,
                                                           verbose=True)

            options = {"optimizer": optimizer, "lr_scheduler": lr_scheduler,
                       "max_epochs": 2, "device": "cpu", "iterations": 1,
                       "sample_loader": None,
                       "num_support_tr": 6}

            train_engine.train(options=options)
        self.assertEqual(str(error.value), "Sample loader has not been specified")

    def test_trainer_8(self):

        init_seed(options={"seed": 0})

        # learning rate scheduler step
        lr_scheduler_step = 15
        num_support_tr = 6
        num_query_tr = 12
        num_samples = num_support_tr + num_query_tr
        # number of random classes per episode for training
        # this should be equal or less than the unique number
        # of classes in the dataset
        classes_per_it = 3
        iterations = 10

        proto_net = ProtoNetTUF(encoder=linear(in_features=2, out_features=3))
        train_engine = TrainEngine(model=proto_net)

        # optimizer to be used for learning
        optimizer = optim.Adam(params=proto_net.parameters(),
                               lr=0.1, weight_decay=0.001)

        # how to reduce the learning rate
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.01,
                                                       step_size=lr_scheduler_step,  verbose=True)

        train_loader = TUFDataset(filename=Path("./test_data/train_data.csv"), dataset_type="train")
        sampler = BatchSampler(labels=train_loader.labels, classes_per_it=classes_per_it,
                               num_samples=num_samples, iterations=iterations, mode="train")

        dataloader = torch.utils.data.DataLoader(train_loader, batch_sampler=sampler)
        options = {"optimizer": optimizer, "lr_scheduler": lr_scheduler,
                   "max_epochs": 1, "device": "cpu", "sample_loader": dataloader,
                   "iterations": iterations, "num_support_tr": num_support_tr}

        train_engine.train(options=options)


if __name__ == '__main__':
    unittest.main()