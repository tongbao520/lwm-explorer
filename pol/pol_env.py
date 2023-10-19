
import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
STEPS = [LEFT, DOWN, RIGHT, UP]
OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}


def step_grid(cur, d, size):
    x, y = cur
    if d == LEFT:
        x -= 1
    elif d == RIGHT:
        x += 1
    elif d == UP:
        y -= 1
    elif d == DOWN:
        y += 1
    if x < 0 or y < 0 or x >= size or y >= size:
        return cur
    return (x, y)


def gen_labyrinth(size, np_random):
    edges = np.zeros((size, size, 4), dtype=bool)
    visit = np.zeros((size, size), dtype=bool)
    stack = [(0, 0)]