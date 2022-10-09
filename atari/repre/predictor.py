import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim
from dqn.prepare_obs import prepare_obs


class PredictorModel(nn.Module):
    def __init__(self, num_action, fstack, emb_size, rnn_size):
        super(PredictorModel, self).__init__()
        self.fstack = fstack
        self.num_action = num_action