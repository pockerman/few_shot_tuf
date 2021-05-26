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

    def __init__(self, model=None):

        self._state = {"epoch": 0,
                       "model": model,
                       "stop": False}

        # various hooks to apply
        self._hooks = {}

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

    def train(self, options):

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
        optimizer = options["optimizer"]
        tr_dataloader = options["sample_loader"]

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0

        while self._state["epoch"] < max_epochs:
            print('> Epoch: {0}'.format(self._state["epoch"]))

            # Let the model know that we are training
            # see the thread: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            # this does not train the model as it may be implied but
            # it informs the model that it undergoes training
            self._state['model'].train()

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

                #print("> X shape {0}".format(X.shape))
                #print("> y shape {0}".format(y.shape))

                X.to(options['device'])
                y.to(options['device'])

                # apply the model to predict the label
                model_output = self._state['model'](X)

                #print("> model out {0}".format(model_output))

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

                train_loss.append(loss.item())
                train_acc.append(acc.item())

            avg_loss = np.mean(train_loss[-options["iterations"]:])
            avg_acc = np.mean(train_acc[-options["iterations"]:])
            print('Average Train Loss: {}, Average Train Acc: {}'.format(avg_loss, avg_acc))
            options["lr_scheduler"].step()
            self._state["epoch"] += 1