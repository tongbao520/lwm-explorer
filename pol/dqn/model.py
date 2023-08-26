import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DQN(nn.Module):
    def __init__(self, rnn_size, device="cuda"):
        super(DQN, self).__init__()
        self.device = device
        self.rnn_size = rnn_size

        self.encoder = nn.Sequential(nn.Linear(4 + 4 + 1, 32), nn.ReLU())
        # self.rnn = nn.GRUCell(32, self.rnn_size)
        self.rnn = nn.GRU(32, self.rnn_size)
        self.adv = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
         