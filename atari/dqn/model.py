import torch
import torch.nn as nn
from torch.nn.functional import one_hot, relu
from dqn.prepare_obs import prepare_obs


def mnih_cnn(size