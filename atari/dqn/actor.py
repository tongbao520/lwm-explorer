
from itertools import count
import torch
import numpy as np
from common.timer import timer_log


def actor_iter(env, model, predictor, warmup, eps=None):
    minstep = int(warmup / env.num_envs)
    hx = (
        torch.zeros(env.num_envs, 512, device=model.device)
        if model is not None
        else None
    )
    hx_pred = None
    step = {"obs": env.reset()}

    timer = timer_log(100)
    next(timer)
    if eps is None:
        eps = (0.4 ** torch.linspace(1, 8, env.num_envs))[..., None]
    mean_reward, mean_len = [], []
    log = {}

    for n_iter in count():
        full_step = yield step, hx, log

        timer.send("actor/action")
        action = torch.randint(env.action_space.n, (env.num_envs, 1))
        if n_iter >= minstep:
            with torch.no_grad():
                _, ri, hx_pred = predictor.get_error(full_step, hx_pred)
                full_step = full_step[1:]
                full_step["reward"] += ri

                qs, hx = model(**full_step, hx=hx)
                action_greedy = qs[0].argmax(1)[..., None].cpu()
            x = torch.rand(env.num_envs, 1) > eps