from functools import partial
import torch
from common.pad_dim import pad_dim


def vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inv_vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (
        (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps))) - 1) / (2 * eps)) ** 2
        - 1
    )


def n_step_bellman_target(reward, done, q, gamma, n_step):
    mask = 1 - pad_dim(done, dim=0, size=n_step - 1)
    reward = pad_dim(reward, dim=0, size=n_step - 1)
    for 