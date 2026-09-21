import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


class Cfg(dict):
    """Dict with attribute access (nested)."""
    def __getattr__(self, k):
        v = self[k]
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path):
    with open(path) as f:
        return Cfg(yaml.safe_load(f))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(cfg):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)
