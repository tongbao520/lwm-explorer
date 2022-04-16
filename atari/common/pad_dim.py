import torch.nn.functional as F


def pad_dim(x, dim, size=1, value=0, left=False):
    p = [0] *