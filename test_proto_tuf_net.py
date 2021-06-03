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
def test(configuration: dict) -> None:


    device = configuration['device']

    if device == 'gpu' and not torch.cuda.is_available():
        print("{0} You specified CUDA as device but PyTorch configuration does not support CUDA".format(WARNING))
        print("{0} Setting device to cpu".format(WARNING))
        configuration['device'] = 'cpu'

    # initialize seed for random generation utilities
    init_seed(options=configuration)

    test_model_path = Path(configuration["save_model_path"] + "/" +
                           configuration["model_name"] + "/" +
                           configuration["test_model"])

    model = ProtoNetTUF.build_network(encoder=linear_with_softmax(in_features=configuration["in_features"],
                                                                  out_features=len(configuration["classes"])),
                                      options=configuration)

    model.load_state_dict(torch.load(test_model_path))

    train_dataset = TUFDataset(filename=Path(configuration["test_dataset"]),
                               dataset_type="test",
                               classes=configuration["classes"])

    print(f"{INFO} Test dataset size {len(train_dataset)} ")

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

    '''
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
    '''


if __name__ == '__main__':

    print("{0} Training prototypical network".format(INFO))
    config_filename = Path("./config.json")
    configuration = read_json(filename=config_filename)
    test(configuration=configuration)
