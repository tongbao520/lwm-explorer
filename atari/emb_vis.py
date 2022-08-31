
import sys
import torch
import time
from gym.envs.atari.atari_env import ACTION_MEANING
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from atari import make_vec_envs
from dqn.model import mnih_cnn


ACTION_ID = {v: k for k, v in ACTION_MEANING.items()}
KEY2ACTION = {
    "w": ACTION_ID["UP"],
    "s": ACTION_ID["DOWN"],
    "a": ACTION_ID["LEFT"],
    "d": ACTION_ID["RIGHT"],
    "f": ACTION_ID["FIRE"],
    "i": ACTION_ID["UPFIRE"],
    "k": ACTION_ID["DOWNFIRE"],
    "j": ACTION_ID["LEFTFIRE"],
    "l": ACTION_ID["RIGHTFIRE"],
    "u": ACTION_ID["UPLEFTFIRE"],
    "o": ACTION_ID["UPRIGHTFIRE"],
}


def convert_key(a):
    return KEY2ACTION.get(chr(a), ACTION_ID["NOOP"])


def key_press(key, mod):
    global cur_action, restart, pause
    if key == 0xFF0D:
        restart = True
    if key == 32:
        pause = not pause
    cur_action = convert_key(key)


def key_release(key, mod):
    global cur_action
    a = convert_key(key)
    if cur_action == a:
        cur_action = 0

