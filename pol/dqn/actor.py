from itertools import count
import torch
import numpy as np
from common.timer import timer_log


def actor_iter(env, model, predi