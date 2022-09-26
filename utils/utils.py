import random
import json
import os

import numpy as np
import torch
import pickle


def set_seed(random_seed: int = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def dump_config(args, save_dir):
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f)


def read_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def setup_path(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    os.makedirs(os.path.join(path, "board"), exist_ok=True)
