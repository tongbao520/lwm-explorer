import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DQN(nn.