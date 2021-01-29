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
from tuf_dataset import TUFDataset

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
    init_seed(manual_seed=options["seed"])

    # the model to train
    model = PrototypicalNetTUF.build_network(options=options)

    # initialize the optimizer
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=options["learning_rate"],
                             weight_decay=options["weight_decay"])

    # initialize scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                    gamma=options["lr_scheduler_gamma"],
                                                    step_size=options["lr_scheduler_step"])

    tr_tuf_dataset = TUFDataset(filename="data/train/proto/proto_support.csv", dataset_type="support")
    tr_query_dataset = TUFDataset(filename="data/train/proto/proto_query.csv", dataset_type="query")

    support_sampler = EpisodicTask.init_sampler(labels=tr_tuf_dataset.labels,
                                                mode='train', options=options)

    query_sampler = EpisodicTask.init_sampler(labels=tr_query_dataset.labels,
                                              mode='query', options=options)

    tr_dataloader = init_dataloader(dataset=tr_tuf_dataset, sampler=support_sampler)
    query_dataloader = init_dataloader(dataset=tr_tuf_dataset, sampler=query_sampler)

    engine_init_state = {"model": model,
                         "task_loader": {"xs": tr_dataloader, "xq": query_dataloader},
                         "optimization_method": optim,
                         "lr_scheduler": lr_scheduler,
                         "max_epochs": options["max_epochs"],
                         "device": options["device"]}

    engine = TrainEngine(init_state=engine_init_state)

    # train the model
    engine.train(options=options)


if __name__ == '__main__':
    print("{0} Training prototypical ne")
    train()