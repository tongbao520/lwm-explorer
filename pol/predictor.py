import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim


class PredictorModel(nn.Module):
    def __init__(self, rnn_size):
        super(PredictorModel, self).__init__()
        self.rnn_size = rnn_size
        s