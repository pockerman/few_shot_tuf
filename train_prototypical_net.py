from pathlib import Path
import torch
from train_engine import TrainEngine
from prototypical_net_tuf import ProtoNetTUF
from utils import timefn
from utils import INFO, WARNING
from utils import read_json, init_seed
from tuf_dataset import TUFDataset
from encoders import linear, linear_with_softmax
from batch_sampler import BatchSampler


@timefn
def train(configuration: dict) -> None:

    device = configuration['device']

    if device == 'gpu' and not torch.cuda.is_available():
        print("{0} You specified CUDA as device but PyTorch configuration does not support CUDA".format(WARNING))
        print("{0} Setting device to cpu".format(WARNING))
        configuration['device'] = 'cpu'

    num_samples = configuration["num_support_tr"] + configuration["num_query_tr"]

    # initialize seed for random generation utilities
    init_seed(options=configuration)

    # the model to train
    model = ProtoNetTUF.build_network(encoder=linear_with_softmax(in_features=configuration["in_features"],
                                                                  out_features=configuration["out_features"]),
                                      options=configuration)

    # initialize the optimizer
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=configuration["optimizer"]["lr"],
                             weight_decay=configuration["optimizer"]["weight_decay"])

    # initialize scheduler for learning rate decay
    # Decays the learning rate of each parameter group by gamma every step_size epochs.
    # Notice that such decay can  happen simultaneously with other changes
    # to the learning rate from outside this scheduler.
    # When last_epoch=-1, sets initial lr as lr.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                    gamma=configuration["lr_scheduler"]["gamma"],
                                                    step_size=configuration["lr_scheduler"]["step_size"])

    train_loader = TUFDataset(filename=Path(configuration["train_dataset"]), dataset_type="train")
    sampler = BatchSampler(labels=train_loader.labels,
                           classes_per_it=configuration["classes_per_it"],
                           num_samples=num_samples,
                           iterations=configuration["iterations"], mode="train")

    dataloader = torch.utils.data.DataLoader(train_loader, batch_sampler=sampler)
    engine = TrainEngine(model=model)

    # train the model
    engine.train(options=TrainEngine.build_options(optimizer=optim, lr_scheduler=lr_scheduler,
                                                   max_epochs=configuration["max_epochs"],
                                                   iterations=configuration["iterations"],
                                                   device=configuration["device"],
                                                   sample_loader=dataloader,
                                                   num_support_tr=configuration["num_support_tr"]))


if __name__ == '__main__':
    print("{0} Training prototypical network".format(INFO))
    config_filename = Path("./config.json")
    configuration = read_json(filename=config_filename)
    train(configuration=configuration)
