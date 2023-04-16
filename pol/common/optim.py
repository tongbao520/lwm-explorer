from dataclasses import dataclass
from typing import List
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


@dataclass
class ParamOptim:
    params: List[torch.Tensor]
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad: float = None

    def __post_init__(self):
        self.optim = Adam(self.params, lr=self.lr, eps=self.eps)

    def scale_lr(self, k):
        for pg in self.optim.param_groups:
          