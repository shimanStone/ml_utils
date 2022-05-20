#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 10:42
# @Author  : shiman
# @File    : model_utils.py
# @describe:

import torch
import torch.nn as nn

astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)


class Accumulator:
    """accumalating sums over n variables"""
    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] *len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def get_accuracy(y_hat, y):
    """compute the number of correct predictions"""

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = astype(y_hat, y.dtype) == y
    sum = reduce_sum(astype(cmp, y.dtype))

    return float(sum)


def evaluate_accuracy(net, data_iter, device):
    """compute the accuracy for a model on a dataset"""

    if isinstance(net, torch.nn.Module):
        net = net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(get_accuracy(net(y), y), y.size())
    return metric[0]/metric[1]


def init_weight(net, type='normal'):
    """initial net weight"""
    def init_func(m):
        pass

    net.apply(init_func)

    return net