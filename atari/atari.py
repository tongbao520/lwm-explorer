import torch
from gym.spaces.box import Box
from baselines import bench
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.atari_wrappers import wrap_deepmind, make_atari
from baselines.common.vec_env import VecEnvWrapper


def make_vec_envs(name, num, seed=0, max_ep_len=100000):
    def make_env(rank):
        def _thunk():
            full_name = f"{name}NoFrameskip-v4"
            env = make_atari(full_name, max_episode_steps=max_ep_len)
            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            env = wrap_deepmind(env, episode_life=True, clip_rewa