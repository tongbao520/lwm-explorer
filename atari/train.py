import argparse
from tqdm import trange
import torch
import wandb

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from atari import make_vec_envs
from dqn import actor_iter, Learner, DQN
from repre.w_mse import WMSE
from repre.predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MontezumaRevenge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ri_scale", type=float, default=1)
    p = parser.parse_args()
    cfg = load_cfg(p.cfg)
    cfg.update(vars(p))
    wandb.init(project="lwm", config=cfg)

    num_env = cfg["agent"]["actors"]
    fstack = cfg["agent"]["frame_stack"]
    envs = make_vec_envs(cfg["env"], num_env, cfg["seed"], cfg["train"]["max_ep_len"])

    buffer = Buffer(
        num_env=num_env,
        maxlen=int(cfg["buffer"]["size"] / num_env),
        obs_shape=envs.observation_space.shape,
        device=cfg["buffer"]["device"],
    )
    model = DQN(envs.action_space.n, fstack).cuda().train()
    wmse = WMSE(buffer, cfg)
    pred = Predictor(buffer, wmse.encoder, envs.action_space.n, cfg)
    learner = Learner(model, buffer, pred, cfg)
    actor = actor_iter(
        envs, model, pred, cfg["buffer"]["warmup"], eps=cfg["agent"].get("eps")
    )

    start_train = int(cfg["buffer"]["warmup"] / num_env)
    log_every = cfg["train"]["log_every"]