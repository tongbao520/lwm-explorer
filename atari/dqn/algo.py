from functools import partial
import torch
from common.pad_dim import pad_dim


def vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inv_vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (
        (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 +