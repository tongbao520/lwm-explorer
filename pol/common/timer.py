from time import time
from collections import defaultdict
import numpy as np


def timer_log(num_iter=1000):
    log = {}
    mean_t = defaultdict(list)
    t = mark = None
    while True:
        prev_t, prev_mark = t, mark
        mark = yield log
   