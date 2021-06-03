from pathlib import Path
from utils import read_json
from train_proto_tuf_net import train
from test_proto_tuf_net import test
from utils import INFO, WARNING

if __name__ == '__main__':
    print("{0} Training prototypical network".format(INFO))
    config_filename = Path("./config.json")
    configuration = read_json(filename=config_filename)

    if configuration["mode"] == "train":
        train(configuration=configuration)
    elif configuration["mode"] == "test":
        test(configuration=configuration)
    else:
        raise ValueError(f"Unknown mode {configuration['mode']}")