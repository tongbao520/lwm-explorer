
from copy import deepcopy
from functools import partial
import random
import torch

from common.optim import ParamOptim
from dqn.algo import get_td_error


class Learner:
    def __init__(self, model, buffer, predictor, cfg):
        model_t = deepcopy(model)
        model_t = model_t.cuda().eval()
        self.model, self.model_t = model, model_t
        self.buffer = buffer
        self.predictor = predictor
        self.optim = ParamOptim(params=model.parameters(), **cfg["optim"])

        self.batch_size = cfg["agent"]["batch_size"]
        self.unroll = cfg["agent"]["unroll"]
        self.unroll_prefix = cfg["agent"]["burnin"] + 1
        self.sample_steps = self.unroll_prefix + self.unroll

        self.target_tau = cfg["agent"]["target_tau"]
        self.td_error = partial(get_td_error, model=model, model_t=model_t, cfg=cfg)
        self.add_ri = cfg["add_ri"]

    def _update_target(self):
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * (1.0 - self.target_tau) + s.data * self.target_tau)

    def loss_uniform(self):
        if len(self.buffer) < self.buffer.maxlen:
            no_prev = set(range(self.sample_steps))
        else:
            no_prev = set(
                (self.buffer.cursor + i) % self.buffer.maxlen
                for i in range(self.sample_steps)
            )
        all_idx = list(set(range(len(self.buffer))) - no_prev)
        idx0 = torch.tensor(random.choices(all_idx, k=self.batch_size))
        idx1 = torch.tensor(
            random.choices(range(self.buffer.num_env), k=self.batch_size)
        )
        batch = self.buffer.query(idx0, idx1, self.sample_steps)
        loss_pred, ri, _ = self.predictor.get_error(batch, update_stats=True)
        if self.add_ri: