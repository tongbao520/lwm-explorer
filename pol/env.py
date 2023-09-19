
import torch
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.wrappers import TimeLimit
from pol_env import PolEnv


def make_vec_envs(num, size, seed=0, max_ep_len=1000):
    def make_env(rank):
        def _thunk():
            env = PolEnv(size)
            env = TimeLimit(env, max_episode_steps=max_ep_len)
            env.seed(seed + rank)