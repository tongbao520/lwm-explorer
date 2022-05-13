from functools import partial
import torch
from common.pad_dim import pad_dim


def vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(tor