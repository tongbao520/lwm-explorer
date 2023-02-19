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
    cfg 