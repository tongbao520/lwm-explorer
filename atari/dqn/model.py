import torch
import torch.nn as nn
from torch.nn.functional import one_hot, relu
from dqn.prepare_obs import prepare_obs


def mnih_cnn(size_in, size_out):
    return nn.Sequential(
        nn.Conv2d(size_in, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2