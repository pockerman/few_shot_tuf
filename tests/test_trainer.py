import unittest
import pytest

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from training_engine import TrainEngine
from prototypical_net_tuf import ProtoNetTUF

class TestTrainEngine(unittest.TestCase):

    def test_construction(self):
        train_engine = TrainEngine()

    def test_train_1(self):

        with pytest.raises(AssertionError) as error:
            train_engine = TrainEngine(model=None)
            train_engine.train(options=None)

        self.assertEqual(str(error.value), "Model has not been specified")

    def test_train_2(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF()
            train_engine = TrainEngine(model=proto_net)
            train_engine.train(options=None)

        self.assertEqual(str(error.value), "Training options not specified")

    def test_train_3(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF()
            train_engine = TrainEngine(model=proto_net)

            options = {"lr_scheduler": None}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Learning scheduler has not been specified")

    def test_train_4(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF()
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)

            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01,
                                                           step_size=0.5,
                                                           verbose=True)

            options = {"optimizater": None, "lr_scheduler": lr_scheduler}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Optimizer has not been specified")

    def test_train_5(self):

        with pytest.raises(AssertionError) as error:

            proto_net = ProtoNetTUF()
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)
            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01,
                                                           step_size=0.5,
                                                           verbose=True)

            options = {"optimizater": optimizer, "lr_scheduler": lr_scheduler,
                       "max_epochs": 0}
            train_engine.train(options=options)

        self.assertEqual(str(error.value), "Invalid number of max epochs")

    def test_trainer_6(self):
        with pytest.raises(AssertionError) as error:
            proto_net = ProtoNetTUF()
            train_engine = TrainEngine(model=proto_net)

            # optimizer to be used for learning
            optimizer = optim.Adam(params=proto_net.parameters(),
                                   lr=0.1, weight_decay=0.01)
            # how to reduce the learning rate
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           gamma=0.01,
                                                           step_size=0.5,
                                                           verbose=True)

            options = {"optimizater": optimizer, "lr_scheduler": lr_scheduler,
                       "max_epochs": 2, "device": "cpu", "sample_loader": None}
            train_engine.train(options=options)
        self.assertEqual(str(error.value), "Sample loader has not been specified")


    def test_trainer_7(self):

        proto_net = ProtoNetTUF()
        train_engine = TrainEngine(model=proto_net)

        # optimizer to be used for learning
        optimizer = optim.Adam(params=proto_net.parameters(),
                               lr=0.1, weight_decay=0.01)
        # how to reduce the learning rate
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       gamma=0.01,
                                                       step_size=0.5,
                                                       verbose=True)

        options = {"optimizater": optimizer, "lr_scheduler": lr_scheduler,
                   "max_epochs": 2, "device": "cpu", "sample_loader": None}
        train_engine.train(options=options)


if __name__ == '__main__':
    unittest.main()