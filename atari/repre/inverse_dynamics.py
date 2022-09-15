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
    lr: float = 5e-4
    frame_stack: int = 1
    device: str = "cuda"

    def __post_init__(self):
        self.encoder = mnih_cnn(self.frame_stack, self.emb_size)
        self.encoder = self.encoder.to(self.device).train()
        self.clf = nn.Sequential(
            nn.Linear(self.emb_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_action),
        )
        self.clf = self.clf.to(self.device).train()
        params = chain(self.encoder.parameters(), self.clf.parameters())
        self.optim