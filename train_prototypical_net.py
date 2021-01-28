import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

from training_engine import TrainEngine
from prototypical_net_tuf import PrototypicalNetTUF
from episodic_task import EpisodicTask
from utils import timefn
from utils import INFO
from utils import read_json
from train_helpers import init_seed
from train_helpers import init_dataloader

@timefn
def train():

    # the device to use
    device = 'cpu'
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("{0} Using CPU device".format(INFO))

    options = read_json(filename="train_config.json")

    # initialize seed for random generation utilities
    init_seed(manual_seed=options.seed)

    # the model to train
    model = PrototypicalNetTUF.build_network(options=options)

    # initialize the optimizer
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=options.learning_rate)

    # initialize scheduler
    torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                    gamma=options.lr_scheduler_gamma,
                                    step_size=options.lr_scheduler_step)

    sampler = EpisodicTask.init_sampler(labels=None, mode='train', options=options)

    tr_dataloader = init_dataloader(dataset=None, sampler=sampler)

    engine_init_state = {"model": model,
                         "task_loader": EpisodicTask(n_episodes=10, n_way=6, n_shot=10),
                         "optimization_method": getattr(optim, "Adam"),
                         "optimization_config": {'lr': 0.01,
                                                 'weight_decay': 0.0}}

    engine = TrainEngine(init_state=engine_init_state)

    engine.train()


if __name__ == '__main__':
    print("{0} Training prototypical ne")
    train()