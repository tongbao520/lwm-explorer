import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def