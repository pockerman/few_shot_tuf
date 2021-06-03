import json
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
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

    dirs = os.listdir(configuration["save_model_path"])

    if configuration["model_name"] in dirs:
        raise ValueError(f"Directory {configuration['model_name']} exists")

    # create directory if it doesnt exist
    output_path = Path(configuration["save_model_path"] + "/" + configuration["model_name"])

    # create the output directory
    os.mkdir(path=output_path)

    configuration["save_model_path"] = str(output_path)

    with open(output_path / "config.json", 'w', newline="\n") as fh:
        # save the configuration in the output
        json.dump(configuration, fh)

    device = configuration['device']

    if device == 'gpu' and not torch.cuda.is_available():
        print("{0} You specified CUDA as device but PyTorch configuration does not support CUDA".format(WARNING))
        print("{0} Setting device to cpu".format(WARNING))
        configuration['device'] = 'cpu'

    # initialize seed for random generation utilities
    init_seed(options=configuration)

    # the model to train
    model = ProtoNetTUF.build_network(encoder=linear_with_softmax(in_features=configuration["in_features"],
                                                                  out_features=len(configuration["classes"])),
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
                           iterations=configuration["iterations"],
                           mode="train")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

    # options for the training engine
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

        print(f"{INFO} Number of samples validation {num_samples_validation}")

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
    engine = TrainEngine(model=model)
    engine.train(options=options)

    engine_state = engine.state

    x = [epoch for epoch in range(configuration["max_epochs"])]

    train_loss = engine_state["average_train_loss"]
    validation_loss = engine_state["average_validation_loss"]

    plt.plot(x, train_loss, 'r*', label="Train loss")
    plt.plot(x, validation_loss, 'bo', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend(loc="upper right")
    plt.title("Train vs Validation loss. $\eta=${0}, Iterations/epoch {1}".format(configuration["optimizer"]["lr"],
                                                                                  configuration["iterations"]))
    plt.savefig(Path(configuration["save_model_path"] + "/" + "train_validation_loss.png"))
    plt.close()

    train_acc = engine_state["average_train_acc"]
    validation_acc = engine_state["average_validation_acc"]

    plt.plot(x, train_acc, 'r*', label="Train accuracy")
    plt.plot(x, validation_acc, 'bo', label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.legend(loc="upper right")
    plt.title("Train vs Validation accuracy. $\eta=${0}, Iterations/epoch {1}".format(configuration["optimizer"]["lr"],
                                                                                  configuration["iterations"]))
    plt.savefig(Path(configuration["save_model_path"] + "/" + "train_validation_accuracy.png"))


if __name__ == '__main__':

    print("{0} Training prototypical network".format(INFO))
    config_filename = Path("./config.json")
    configuration = read_json(filename=config_filename)
    train(configuration=configuration)
