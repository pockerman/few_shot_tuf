"""
Training engine for training networks. The implementation
of this class is largely based on the Engine class from
https://github.com/jakesnell/prototypical-networks
"""

import numpy as np
from tqdm import tqdm


class TrainEngine(object):
    """
    Training engine to act as a wrapper
    for training
    """

    def __init__(self, init_state=None):

        if init_state is None:
            # state of the engine. After training
            # this parameter will contain the trained model
            self._state = {"epoch": 0,
                           "max_epochs": 0,
                           "model": None,
                           "stop": False,
                           "optimizer": None,
                           "optimization_method": None,
                           "task_loader": None,
                           "decice": 'cpu',
                           "lr_scheduler": None}
        else:
            self._state = init_state

            if "epoch" not in self._state:
                self._state["epoch"] = 0

        # various hooks to apply
        self._hooks = {}

    @property
    def state(self):
        return self._state

    @property
    def max_epochs(self):
        return self._state["max_epochs"]

    @max_epochs.setter
    def max_epochs(self, value):
        self._state["max_epochs"] = value

    @property
    def optimizer(self):
        return self._state["optimization_method"]

    @optimizer.setter
    def optimizer(self, value):
        self._state["optimization_method"] = value

    @property
    def task_loader(self):
        return self._state["task_loader"]

    @task_loader.setter
    def task_loader(self, value):
        self._state["task_loader"] = value

    def train(self, options):

        assert self._state['model'] is not None, "Model has not been specified"
        assert self._state["lr_scheduler"] is not None, "Learning scheduler has not been specified"
        assert self._state["optimization_method"] is not None, "Optimizer has not been specified"
        assert "max_epochs" in self._state, "Maximum number of epochs has not been specified"
        assert self._state["max_epochs"] > 0, "Invalid number of max epochs"
        assert "device" in self._state, "Device has not been specified"

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0

        while self._state["epoch"] < self._state["max_epochs"]: #and not self._state["stop"]:
            print("Training epoch: {0}".format(self._state["epoch"]))

            # Let the model know that we are training
            # see the thread: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            # this does not train the model as it may be implied but
            # it informs the model that it undergoes training
            self._state['model'].train()

            # loop over the sample supplied by the sample loader
            # for the current epoch
            for sample in tqdm(self._state['task_loader']['xs'], desc="Epoch {:d} train".format(self._state['epoch'] + 1)):
                # zero the optimizer gradient
                # that is zero the parameter gradients
                # gradient buffers had to be manually set to zero using optimizer.zero_grad().
                # because gradients are accumulated in the Backprop step
                self._state['optimization_method'].zero_grad()

                X, y = sample
                X.to(self._state['device'])
                y.to(self._state['device'])

                model_output = self._state['model'](X)
                loss, acc = self._state['model'].loss_fn(model_output, target=y,
                                                         n_support=options.num_support_tr)

                # backward propagate
                loss.backward()

                self._state['optimization_method'].step()

                train_loss.append(loss.item())
                train_acc.append(acc.item())

            avg_loss = np.mean(train_loss[-options.iterations:])
            avg_acc = np.mean(train_acc[-options.iterations:])
            print('Average Train Loss: {}, Average Train Acc: {}'.format(avg_loss, avg_acc))
            self._state["lr_scheduler"].step()