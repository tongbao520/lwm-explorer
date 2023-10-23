import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim


class PredictorModel(nn.Module):
    def __init__(