import argparse
from tqdm import trange
import torch
import wandb

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from atari import make_vec_envs
from dqn import actor_iter, Learner, DQN
from repre.w_mse import WMSE
from repre.predictor import Predicto