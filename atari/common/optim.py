from dataclasses import dataclass
from typing import List
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


@dataclass
class ParamOptim:
    params: List[torch.Tensor]
    lr: float = 1e