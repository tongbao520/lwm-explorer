
from copy import deepcopy
from functools import partial
import random
import torch

from common.optim import ParamOptim
from dqn.algo import get_td_error
from dqn.sampler import Sampler


def tde_to_prior(x, eta=0.9):
    return (eta * x.max(0).values + (1 - eta) * x.mean(0)).detach().cpu()


class Learner:
    def __init__(self, model, buffer, predictor, cfg):
        num_env = cfg["agent"]["actors"]
        model_t = deepcopy(model)
        model_t = model_t.cuda().eval()
        self.model, self.model_t = model, model_t
        self.buffer = buffer
        self.predictor = predictor
        self.optim = ParamOptim(params=model.parameters(), **cfg["optim"])

        self.batch_size = cfg["agent"]["batch_size"]
        self.unroll = cfg["agent"]["unroll"]