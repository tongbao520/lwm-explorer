import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DQN(nn.Module):
    def __init__(self, rnn_size, device="cuda"):
        super(DQN, self).__init__()
        self.devi