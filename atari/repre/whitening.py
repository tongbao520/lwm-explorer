import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats