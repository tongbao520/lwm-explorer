from time import time
from collections import defaultdict
import numpy as np


def timer_log(num_iter=1000):
    log = {}
    mean_t = defaultdict(l