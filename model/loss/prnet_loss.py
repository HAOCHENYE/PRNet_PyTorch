# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.19
    @readme: The implementation of PRNet Network Loss.
"""

import os
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from .builder import FACELOSS

def preprocess(mask):
    """
    :param mask: grayscale of mask.
    :return:
    """
    mask[mask > 0] = mask[mask > 0] / 16
    mask[mask == 15] = 16
    mask[mask == 7] = 8
    # for i in mask:
    #     for j in i:
    #         if j not in tmp.keys():
    #             tmp[j] = 1
    #         else:
    #             tmp[j] += 1
    # print(tmp)
    # {0: 21669, 3: 33223, 4: 10429, 8: 147, 16: 68}

    return mask

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _fspecial_gauss(window_size, sigma=1.5):
    # Function to mimic the 'fspecial' gaussian MATLAB function.
    coords = np.arange(0, window_size, dtype=np.float32)
    coords -= (window_size - 1) / 2.0

    g = coords ** 2
    g *= (-0.5 / (sigma ** 2))
    g = np.reshape(g, (1, -1)) + np.reshape(g, (-1, 1))
    g = torch.from_numpy(np.reshape(g, (1, -1)))
    g = torch.softmax(g, dim=1)
    g = g / g.sum()
    return g


# 2019.05.26. butterworth filter.
# ref: http://www.cnblogs.com/laumians-notes/p/8592968.html
def butterworth(window_size, sigma=1.5, n=2):
    nn = 2 * n
    bw = torch.Tensor([1 / (1 + ((x - window_size // 2) / sigma) ** nn) for x in range(window_size)])
    return bw / bw.sum()


def create_window(window_size, channel=3, sigma=1.5, gauss='original', n=2):
    if gauss == 'original':
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.tensor(_2D_window.expand(channel, 1, window_size, window_size).clone().detach().contiguous())
        return window
    elif gauss == 'butterworth':
        _1D_window = butterworth(window_size, sigma, n).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.tensor(_2D_window.expand(channel, 1, window_size, window_size).clone().detatch().contiguous())
        return window
    else:
        g = _fspecial_gauss(window_size, sigma)
        g = torch.reshape(g, (1, 1, window_size, window_size))
        # 2019.06.05.
        # https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853
        g = tile(g, 0, 3)
        return g

@FACELOSS.register_module()
class WeightMaskLoss(nn.Module):
    """
        L2_Loss * Weight Mask
    """

    def __init__(self, mask_path):
        super(WeightMaskLoss, self).__init__()
        if os.path.exists(mask_path):
            self.mask_ = cv2.imread(mask_path, 0)
            self.mask_ = torch.from_numpy(preprocess(self.mask_)).float()
            self.register_buffer("mask", self.mask_)
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, pred, gt):
        result = torch.mean(torch.pow((pred - gt), 2), dim=1)
        # self.mask = self.mask.to(result.device)
        result = torch.mul(result, self.mask)

        # 1) 官方(不除256*256的话, 数值就太大了...).
        result = torch.sum(result)
        result = result / (self.mask.size(1) ** 2)
        # 2) 一般使用的都是mean.
        # result = torch.mean(result)
        return result

def dfl_ssim(img1, img2, mask, window_size=11, val_range=1, gauss='original'):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    # padd = window_size//2
    padd = 0
    (batch, channel, height, width) = img1.size()
    img1, img2 = torch.mul(img1, mask), torch.mul(img2, mask)

    real_size = min(window_size, height, width)
    window = create_window(real_size, gauss=gauss).to(img1.device)

    # 2019.05.07.
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    num0 = mu1 * mu2 * 2.0
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    den0 = mu1_sq + mu2_sq

    luminance = (num0 + c1) / (den0 + c1)

    num1 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) * 2.0
    den1 = F.conv2d(img1 * img1 + img2 * img2, window, padding=padd, groups=channel)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim_val = torch.mean(luminance * cs, dim=(-3, -2))

    return torch.mean((1.0 - ssim_val) / 2.0)


@FACELOSS.register_module()
class SSIM(torch.nn.Module):

    def __init__(self, mask_path, window_size=11, alpha=0.8, gauss='original'):
        super(SSIM, self).__init__()

        self.window_size = window_size
        self.window = None
        self.channel = None

        self.gauss = gauss
        self.alpha = alpha

        if os.path.exists(mask_path):
            self.mask_ = cv2.imread(mask_path, 0)
            self.mask_ = torch.from_numpy(preprocess(self.mask_)).float()
            self.register_buffer("mask", self.mask_)
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, img1, img2):
        img2 = img2.to(img1.device)
        # self.mask = self.mask.to(img1.device)
        (_, channel, _, _) = img1.size()
        self.channel = channel
        return 10 * dfl_ssim(img1, img2, mask=self.mask, window_size=self.window_size, gauss=self.gauss)

