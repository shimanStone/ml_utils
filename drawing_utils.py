#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/20 14:37
# @Author  : shiman
# @File    : drawing_utils.py
# @describe:

import torch

import matplotlib.pyplot as plt


def show_images(imgs, num_rows, num_cols, scale=1.5, titles=None):
    """展示多个图片"""
    figsize = (num_cols*scale, num_cols*scale)  # 定义展示图片尺寸
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(imgs, axes)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 不展示XY轴刻度
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])


def apply(image, aug, num_rows=2, num_cols=4, scale=1.5):
    """定义图像增广方法"""
    y = [aug(image) for _ in range(num_cols*num_rows)]
    show_images(y, num_rows, num_cols, scale=scale)
