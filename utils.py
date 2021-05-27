import json
from functools import wraps

import time
import os
from pathlib import Path
import torch
import numpy as np

INFO = "INFO: "
WARNING = "WARNING: "

def timefn(fn):
    @wraps(fn)
    def measure(*args, **kwargs):
        time_start = time.perf_counter()
        result = fn(*args, **kwargs)
        time_end = time.perf_counter()
        print("{0} Done. Execution time"
              " {1} secs".format(INFO, time_end - time_start))
        return result

    return measure


def init_seed(options: dict) -> None:
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    torch.cuda.manual_seed(options["seed"])


def read_json(filename: Path) -> dict:

    """
        Read the json configuration file and
        return a map with the config entries
    """
    with open(filename, 'r') as json_file:
        json_input = json.load(json_file)
        return json_input


def to_csv_line(data):
    """
    Convert the data to comma separated string
    """

    if isinstance(data, float):
        return str(data)

    if isinstance(data, int):
        return str(data)

    return ','.join(str(d) for d in data)


def type_converter(data, type_to_convert_to):
    """
    Return the data converted into the type
    denoted by the string type_converter
    """

    if type_to_convert_to == "FLOAT":
        return float(data)
    elif type_to_convert_to == "INT":
        return int(data)
    elif type_to_convert_to == "STRING":
        return str(data)

    raise ValueError("Unknown type_converter={0}".format(type_to_convert_to))


def load_data_file(filename, type_convert):
    """
    Loads a .txt data file into an array. Every
    item is converted according to type_convert
    """
    with open(filename) as file:
        context = file.read()
        size = len(context)
        arraystr= context[1:size-1]
        arraystr = arraystr.split(',')
        region_means = [type_converter(data=item, type_to_convert_to=type_convert) for item in arraystr]
        return region_means


def make_data_array(wga_mu, no_wga_mu, gc, use_ratio, use_gc):
    """
    Using the two data arrays returns an array as pairs.

    If gc array is also supplied and use_gc=True then
    returns a array containing the tripplets.

    If use_ratio=True then it returns the trippler
    [no_wga, wga, (wga + 1) / (no_wga + 1)]

    if use_ratio=True and use_gc=True
    it returns the quatruplet
    [no_wga_val, wga_val, (wga_val + 1) / (no_wga_val + 1), gc_val]
    in this case gc should not be None

    """

    if len(no_wga_mu) != len(wga_mu):
        raise ValueError("Invalid data size")

    data = []

    if use_ratio and use_gc:

        if gc is None or len(gc) == 0:
            raise  ValueError("GC array is either None or empty")

        if len(gc) != len(no_wga_mu):
            raise ValueError("GC array size={0} is not equal to {1}".format(len(gc), len(no_wga_mu)))

        for no_wga_val, wga_val, gc_val in zip(no_wga_mu, wga_mu, gc):
            data.append([no_wga_val, wga_val, (wga_val + 1) / (no_wga_val + 1), gc_val])
    elif use_ratio:
        for no_wga, wga in zip(no_wga_mu, wga_mu):
            data.append([no_wga, wga, (wga + 1) / (no_wga + 1)])
    elif use_gc:

        if gc is None:
            raise ValueError("GC array is None")

        if gc is None or len(gc) != len(no_wga_mu):
            raise ValueError("GC array size={0} is not equal to {1}".format(len(gc), len(no_wga_mu)))

        for no_wga_val, wga_val, gc_val in zip(no_wga_mu, wga_mu, gc):
            data.append([no_wga_val, wga_val, gc_val])
    else:

        for no_wga, wga in zip(no_wga_mu, wga_mu):
            data.append([no_wga, wga])

    return data









