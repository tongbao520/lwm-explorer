from itertools import count
import torch
import numpy as np
from common.timer import timer_log


def actor_iter(env, model, predictor, warmup, eps=None):
    minstep = int(warmup / env.num_envs)
    hx = None  # torch.zeros(env.num_envs, model.rnn_size, device=model.device)
    hx_pred = None
    step = {"obs": env.reset()}

    timer = timer_log(100)
    next(timer)
    if eps is None:
        eps = (0.4 ** torch.linspace(1, 8, env.num_envs))[..., None]
    mean_reward, mean_len = [], []
    log = {}

    for n_iter in count():
        