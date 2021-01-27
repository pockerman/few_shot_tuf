"""
Training engine for training networks. The implementation
of this class is largely based on the Engine class from
https://github.com/jakesnell/prototypical-networks
"""

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
            self._state = {"epoch": 0, "max_epoch": 0,
                           "model": None,
                           "stop": False,
                           "optimizer": None,
                           "optimization_method": None,
                           "optimization_config": None,
                           "task_loader": None}
        else:
            self._state = init_state

            if "optimizer" not in self._state:
                self._state["optimizer"] = self._state['optimization_method'](self._state['model'].parameters(),
                                                                              **self._state['optimization_config'])

        # various hooks to apply
        self._hooks = {}

    @property
    def state(self):
        return self._state

    @property
    def max_epoch(self):
        return self._state["max_epoch"]

    @max_epoch.setter
    def max_epoch(self, value):
        self._state["max_epoch"] = value

    @property
    def optimizer(self):
        return self._state["optimizer"]

    @optimizer.setter
    def optimizer(self, value):
        self._state["optimizer"] = value

    @property
    def task_loader(self):
        return self._state["task_loader"]

    @task_loader.setter
    def task_loader(self, value):
        self._state["task_loader"] = value


    def train(self):

        assert self._state['model'] is None, "Model has not been specified"

        while self._state["epoch"] < self._state["max_epoch"] and not self._state["stop"]:
            print("Training epoch: {0}".format(self._state["epoch"]))

            # Let the model know that we are training
            # see the thread: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            # this does not train the model as it may be implied but
            # it informs the model that it undergoes training
            self._state['model'].train()

            # loop over the sample supplied by the sample loader
            # for the current epoch
            for sample in tqdm(self._state['loader'], desc="Epoch {:d} train".format(self._state['epoch'] + 1)):

                # zero the optimizer gradient
                # that is zero the parameter gradients
                # gradient buffers had to be manually set to zero using optimizer.zero_grad().
                # because gradients are accumulated in the Backprop step
                self._state['optimizer'].zero_grad()

                # backward propagate
                loss.backward()

                self._state['optimizer'].step()

