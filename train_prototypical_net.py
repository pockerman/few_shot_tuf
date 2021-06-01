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

def extend_options_from_config(configuration, options):

    if "model_name" in configuration:
        options["model_name"] = configuration["model_name"]

    if "save_model" in configuration:
        options["save_model"] = configuration["save_model"]

    if "validate" in configuration:
        options["validate"] = configuration["validate"]

    if "validate_dataset" in configuration:
        options["validate_dataset"] = configuration["validate_dataset"]

    if "save_model_path" in configuration:
        options["save_model_path"] = configuration["save_model_path"]

    return options


@timefn
def train(configuration: dict) -> None:

    device = configuration['device']

    if device == 'gpu' and not torch.cuda.is_available():
        print("{0} You specified CUDA as device but PyTorch configuration does not support CUDA".format(WARNING))
        print("{0} Setting device to cpu".format(WARNING))
        configuration['device'] = 'cpu'

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

    train_dataset = TUFDataset(filename=Path(configuration["train_dataset"]),
                               dataset_type="train",
                               classes=configuration["classes"])

    print(f"{INFO} Training dataset size {len(train_dataset)} ")

    # number of samples for training
    # num_support_tr is the number of support points per class
    # num_query_tr is the number of query points per class
    num_samples = configuration["num_support_tr"] + configuration["num_query_tr"]
    sampler = BatchSampler(labels=train_dataset.labels,
                           classes_per_it=len(configuration["classes"]),
                           num_samples=num_samples,
                           iterations=configuration["iterations"], mode="train")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

    engine = TrainEngine(model=model)

    # train the model
    options = TrainEngine.build_options(optimizer=optim, lr_scheduler=lr_scheduler,
                                        max_epochs=configuration["max_epochs"],
                                        iterations=configuration["iterations"],
                                        device=configuration["device"],
                                        sample_loader=dataloader,
                                        num_support_tr=configuration["num_support_tr"])

    options = extend_options_from_config(configuration=configuration, options=options)

    if configuration["validate"]:

        num_support_validation = configuration["num_support_validation"]
        num_query_validation = configuration["num_query_validation"]
        num_samples_validation = num_query_validation + num_support_validation

        print("Number of samples validation ", num_samples_validation)

        validation_dataset = TUFDataset(filename=Path(configuration["validate_dataset"]),
                                        dataset_type="validate",
                                        classes=configuration["classes"])
        print(f"{INFO} Validation dataset size {len(validation_dataset)} ")

        val_sampler = BatchSampler(labels=validation_dataset.labels,
                                   classes_per_it=len(configuration["classes"]),
                                   num_samples=num_samples_validation,
                                   iterations=configuration["iterations"],
                                   mode="validate")

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_sampler=val_sampler)
        options["validation_dataloader"] = validation_dataloader
        options["num_support_validation"] = configuration["num_support_validation"]


    # train the model
    engine.train(options=options)

    #if configuration["save_model"]:
    #    save_path = Path(configuration["save_model_path"] / configuration["model_name"])
    #    torch.save(engine.state["model"].state_dict(), save_path)


if __name__ == '__main__':
    print("{0} Training prototypical network".format(INFO))
    config_filename = Path("./config.json")
    configuration = read_json(filename=config_filename)
    train(configuration=configuration)
