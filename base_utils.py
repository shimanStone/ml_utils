#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/20 15:06
# @Author  : shiman
# @File    : base_utils.py
# @describe:

import time
import numpy as np


class Timer:
    """Record multiple running times"""

    def __init__(self):
        self.times = []

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
