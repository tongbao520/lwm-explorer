import torch
import torch.nn as nn
from torch.nn.functional import one_hot, relu
from dqn.prepare_obs import prepare_obs


def mnih_cnn(size_in, size_out):
    return nn.Sequential(
        nn.Conv2d(size_in, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, size_out),
    )


class DQN(nn.Module):
    def __init__(self, size_out, size_stack, device="cuda"):
        super(DQN, self).__init__()
        self.size_out = size_out
        self.size_stack = size_stack
        self.conv = mnih_cnn(size_stack, 512)
        self.rnn = nn.GRUCell(512 + 1 + size_out, 512)
        self.adv = nn.Seq