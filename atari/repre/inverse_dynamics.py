import random
from itertools import chain
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.functional import cross_entropy, relu
from common.optim import ParamOptim
from dqn.model import mnih_cnn
from dqn.buffer import Buffer
from dqn.prepare_obs import prepare_obs


@dataclass
class IDF:
    buffer: Buffer
    num_action: int
    emb_size: int = 32
    batch_size: int = 256
    lr: float = 5e