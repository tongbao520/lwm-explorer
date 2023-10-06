import argparse
import numpy as np
import torch
import wandb

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from env import make_vec_envs
from dqn import actor_iter, DQN
from predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--add_ri", action="store_true")
    parser.add_argument("--random", action="store_true")
    p = parser.parse_args()
    cfg = load_cfg("default")
    cfg.update(vars(p))
    cfg["env"] = "pol"
    wandb.init(project="lwm", config=cfg)

    num_env = 1
    envs = make_vec_envs(
        num=num_env,
        size=cfg["size"],
        max_ep_len=cfg["train"]["max_ep_len"],
    )
    buffer = Buffer(
        num_env=num_env,
        maxlen=int(cfg["buffer"]["size"] / num_env),
        obs_shape=(4,),
        device=cfg["buffer"]["dev