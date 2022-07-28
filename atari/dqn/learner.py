
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
        self.unroll_prefix = (
            cfg["agent"]["burnin"]
            + cfg["agent"]["n_step"]
            + cfg["agent"]["frame_stack"]
            - 1
        )
        self.sample_steps = self.unroll_prefix + self.unroll
        self.hx_shift = cfg["agent"]["frame_stack"] - 1
        num_unrolls = (self.buffer.maxlen - self.unroll_prefix) // self.unroll

        if cfg["buffer"]["prior_exp"] > 0:
            self.sampler = Sampler(
                num_env=num_env,
                maxlen=num_unrolls,
                prior_exp=cfg["buffer"]["prior_exp"],
                importance_sampling_exp=cfg["buffer"]["importance_sampling_exp"],
            )
            self.s2b = torch.empty(num_unrolls, dtype=torch.long)
            self.hxs = torch.empty(num_unrolls, num_env, 512, device="cuda")
            self.hx_cursor = 0
        else:
            self.sampler = None

        self.target_tau = cfg["agent"]["target_tau"]
        self.td_error = partial(get_td_error, model=model, model_t=model_t, cfg=cfg)

    def _update_target(self):
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * (1.0 - self.target_tau) + s.data * self.target_tau)

    def append(self, step, hx, n_iter):
        self.buffer.append(step)

        if self.sampler is not None:
            if (n_iter + 1) % self.unroll == self.hx_shift:
                self.hxs[self.hx_cursor] = hx
                self.hx_cursor = (self.hx_cursor + 1) % len(self.hxs)

            k = n_iter - self.unroll_prefix
            if k > 0 and (k + 1) % self.unroll == 0:
                self.s2b[self.sampler.cursor] = self.buffer.cursor - 1
                x = self.buffer.get_recent(self.sample_steps)
                hx = self.hxs[self.sampler.cursor]
                with torch.no_grad():
                    loss, _ = self.td_error(x, hx)
                self.sampler.append(tde_to_prior(loss))

                if len(self.sampler) == self.sampler.maxlen: