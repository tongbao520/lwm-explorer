import torch
from gym.spaces.box import Box
from baselines import bench
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.atari_wrappers import wrap_deepmind, make_atari
from baselines.common.vec_env import VecEnvWrapper


def make_vec_