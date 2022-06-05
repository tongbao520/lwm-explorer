from functools import partial
import torch
from common.pad_dim import pad_dim


def vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inv_vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (
        (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps))) - 1) / (2 * eps)) ** 2
        - 1
    )


def n_step_bellman_target(reward, done, q, gamma, n_step):
    mask = 1 - pad_dim(done, dim=0, size=n_step - 1)
    reward = pad_dim(reward, dim=0, size=n_step - 1)
    for i in range(n_step):
        q[i:] *= gamma * mask[: len(mask) - i]
        q[i:] += reward[: len(reward) - i]
    return q[n_step - 1 :]


def get_td_error(batch, hx_start, model, model_t, cfg, need_stat=False):
    n_step = cfg["agent"]["n_step"]
    pf = cfg["agent"]["frame_stack"] - 1  # prefix for frame stack
    burnin = cfg["agent"]["burnin"]
    bellman_target = partial(
        n_step_bellman_target,
        gamma=cfg["agent"]["gamma"],
        n_step=cfg["agent"]["n_step"],
    )

    if burnin > 0:
        with torch.no_grad():
            hx = model(**batch[: burnin + pf], hx=hx_start, only_hx=True)
            hx_target = model_t(**batch[: burnin + 1 + pf], hx=hx_start, only_hx=True)
    else:
        hx = hx_target = None

    qs, _ = model(**batch[burnin:], hx=hx)

    with torch.no_grad():
        qs_target, _ = model_t(**batch[burnin + 1 :], hx=hx_target)

    action = batch["action"][burnin + pf + 1 : -n_step + 1]
    reward = batch["reward"][burnin + pf + 1 : -n_step + 1]
    done = batch["done"][burnin + pf + 1 : -n_step + 1].float()

    q = qs[:-n_step].gather(2, action)
    ns_action = qs[1:].argmax(2)[..., None].detach()
    next_q = qs_target.gather(2, ns_action)
    next_q = inv_vf_rescaling(next_q)
    target_q = bellman_target(reward, done, next_q)
    target_q = vf_rescaling(target_q)
    td_error = (q - target_q).abs()

    if need_stat:
        log = {
            "loss": td_error.mea