"""
Training engine for training networks. The implementation
of this class is largely based on the Engine class from
https://github.com/jakesnell/prototypical-networks
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import torch
from utils import INFO


class TrainEngine(object):
    """
    Training engine to act as a wrapper
    for training
    """

    @staticmethod
    def build_options(optimizer, lr_scheduler, max_epochs,
                      iterations, sample_loader, device,
                      num_support_tr) -> dict:

        options = dict()
        options["lr_scheduler"] = lr_scheduler
        options["optimizer"] = optimizer
        options["max_epochs"] = max_epochs
        options["iterations"] = iterations
        options["sample_loader"] = sample_loader
        options["device"] = device
        options["num_support_tr"] = num_support_tr
        return options

    def __init__(self, model=None):

        self._state = {"epoch": 0,
                       "model": model,
                       "stop": False,
                       "average_train_loss": [],
                       "average_train_acc": [],
                       "average_validation_loss": [],
                       "average_validation_acc": [],
                       "train_loss": [],
                       "train_acc": [],
                       "validation_loss": [],
                       "validation_acc": [],
                       "best_acc": 0.0,
                       "best_model_state": None}

    @property
    def state(self):
        return self._state

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

    def train(self, options: dict) -> None:

        # make sure that engine state is sane
        assert self._state['model'] is not None, "Model has not been specified"

        assert options is not None, "Training options not specified"
        assert options["lr_scheduler"] is not None, "Learning scheduler has not been specified"
        assert options["optimizer"] is not None, "Optimizer has not been specified"
        assert options["max_epochs"] > 0, "Invalid number of max epochs"
        assert options["iterations"] > 0, "Invalid number of iterations per epoch"
        assert options["sample_loader"] is not None, "Sample loader has not been specified"
        assert "device" not in options["device"], "Device has not been specified"

        max_epochs = options["max_epochs"]
        self._state["best_model_state"] = None
        self._state["best_acc"] = 0.0

        self._state["train_loss"] = []
        self._state["train_acc"] = []
        self._state["validation_loss"] = []
        self._state["validation_acc"] = []

        self._state["average_train_loss"] = []
        self._state["average_train_acc"] = []
        self._state["average_validation_loss"] = []
        self._state["average_validation_acc"] = []

        while self._state["epoch"] < max_epochs:

            print('> Epoch: {0}'.format(self._state["epoch"]))

            # Let the model know that we are training
            # see the thread: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            # this does not train the model as it may be implied but
            # it informs the model that it undergoes training
            self._state['model'].train()
            avg_loss, avg_acc = self.train_model(options=options)
            print('Average Train Loss: {}, Average Train Acc: {}'.format(avg_loss, avg_acc))
            self._state["average_train_loss"].append(avg_loss)
            self._state["average_train_acc"].append(avg_acc)

            options["lr_scheduler"].step()
            self._state["epoch"] += 1

            if 'validate' in options and options["validate"]:

                print("{0} Validating model...".format(INFO))
                self._state['model'].eval()
                avg_loss, avg_acc = self.validate_model(options=options)
                postfix = ' (Best)' if avg_acc >= self._state["best_acc"] else ' (Best: {})'.format(self._state["best_acc"])
                print('Avg Validation Loss: {}, Avg Validation Acc: {}{}'.format( avg_loss, avg_acc, postfix))

                self._state["average_validation_loss"].append(avg_loss)
                self._state["average_validation_acc"].append(avg_acc)

                if avg_acc >= self._state["best_acc"]:

                    if "save_model" in options and options["save_model"]:

                        save_path = Path(options["save_model_path"]) /options["model_name"]
                        torch.save(self._state['model'].state_dict(), save_path)
                    self._state["best_acc"] = avg_acc
                    self._state["best_model_state"] = self._state['model'].state_dict()

        if 'validate' in options and options["validate"]:
            save_path = Path(options["save_model_path"]) / "best_model"
            torch.save(self._state['best_model_state'], save_path)

    def train_model(self, options: dict) -> Tuple[float, float]:

        optimizer = options["optimizer"]
        tr_dataloader = options["sample_loader"]

        tr_iter = iter(tr_dataloader)

        # loop over the sample supplied by the sample loader
        # for the current epoch. Here we effectively extract a batch
        for batch in tqdm(tr_iter):

            # zero the optimizer gradient
            # that is zero the parameter gradients
            # gradient buffers had to be manually set to zero using optimizer.zero_grad().
            # because gradients are accumulated in the Backprop step
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            X, y = batch

            X.to(options['device'])
            y.to(options['device'])

            # apply the model to predict the label
            model_output = self._state['model'](X)

            # comput the loss
            loss, acc = self._state['model'].loss_fn(input=model_output, target=y,
                                                     n_support=options["num_support_tr"])
            # backward propagate.
            # This call will compute the
            #  gradient of loss with respect to all Tensors with requires_grad=True.
            loss.backward()

            # # Calling the step function on an
            # Optimizer makes an update to its parameters
            optimizer.step()

            self._state["train_loss"].append(loss.item())
            self._state["train_acc"].append(acc.item())

        avg_loss = np.mean(self._state["train_loss"][-options["iterations"]:])
        avg_acc = np.mean(self._state["train_acc"][-options["iterations"]:])

        return avg_loss, avg_acc

    def validate_model(self, options: dict) -> Tuple[float, float]:

        device = options["device"]
        validation_dataloader = options["validation_dataloader"]
        validation_iter = iter(validation_dataloader)

        for validation_batch in tqdm(validation_iter):
            x, y = validation_batch
            x, y = x.to(device), y.to(device)
            model_output = self._state['model'](x)
            loss, acc = self._state['model'].loss_fn(model_output, target=y,
                                                     n_support=options["num_support_validation"])

            self._state["validation_loss"].append(loss.item())
            self._state["validation_acc"].append(acc.item())

        avg_loss = np.mean(self._state["validation_loss"][-options["iterations"]:])
        avg_acc = np.mean(self._state["validation_acc"][-options["iterations"]:])
        return avg_loss, avg_acc
