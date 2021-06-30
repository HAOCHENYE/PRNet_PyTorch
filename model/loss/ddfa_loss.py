#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import math
from .builder import FACELOSS
import numpy as np
from model.temp_func import _load



def _parse_param_batch(param):
    """Work for both numpy and tensor"""
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:52].view(N, -1, 1)
    alpha_exp = param[:, 52:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp

@FACELOSS.register_module()
class VDCLoss(nn.Module):
    def __init__(self,
                 used_item=dict(param_mean=None,
                                param_std=None,
                                u=None,
                                w_shp=None,
                                w_exp=None,
                                keypoints=None,
                                ),
                 opt_style='all'):
        super(VDCLoss, self).__init__()

        for attr, value in used_item.items():
            assert isinstance(value, torch.Tensor), "used_item should be define in DDFA model!!!"
            self.register_buffer(attr, value)

        self.u_base = self.u[self.keypoints]
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

        self.w_shp_length = self.w_shp.shape[0] // 3
        self.opt_style = opt_style

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def forward_all(self, input, target):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        N = input.shape[0]
        offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward_resample(self, input, target, resample_num=132):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        # resample index
        index = torch.randperm(self.w_shp_length)[:resample_num].reshape(-1, 1)
        keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
        keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

        offset[:, -1] = offsetg[:, -1]

        N = input.shape[0]
        gt_vertex = pg @ (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset
        diff = (gt_vertex - vertex) ** 2
        loss = torch.mean(diff)
        return loss

    def forward(self, input, target):
        if self.opt_style == 'all':
            return self.forward_all(input, target)
        elif self.opt_style == 'resample':
            return self.forward_resample(input, target)
        else:
            raise Exception(f'Unknown opt style: f{opt_style}')


@FACELOSS.register_module()
class WPDCLoss(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self,
                 opt_style='resample',
                 resample_num=132,
                 used_item=dict(param_mean=None,
                                param_std=None,
                                u=None,
                                w_shp=None,
                                w_exp=None,
                                keypoints=None,
                                )):
        super(WPDCLoss, self).__init__()
        for attr, value in used_item.items():
            assert isinstance(value, torch.Tensor), "used_item should be define in DDFA model!!!"
            self.register_buffer(attr, value)

        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)
        self.w_shp_length = self.w_shp.shape[0] // 3
        self.resample_num = resample_num
        self.opt_style = opt_style

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def _calc_weights_resample(self, input_, target_):
        # resample index
        if self.resample_num <= 0:
            keypoints_mix = self.keypoints
        else:
            index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
            keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
            keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

        input = torch.tensor(input_.data.clone(), requires_grad=False)
        target = torch.tensor(target_.data.clone(), requires_grad=False)

        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        input = self.param_std * input + self.param_mean
        target = self.param_std * target + self.param_mean

        N = input.shape[0]

        offset[:, -1] = offsetg[:, -1]

        weights = torch.zeros_like(input, dtype=torch.float)
        tmpv = (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg).view(N, -1, 3).permute(0, 2, 1)

        tmpv_norm = torch.norm(tmpv, dim=2)
        offset_norm = math.sqrt(w_shp_base.shape[0] // 3)

        # for pose
        param_diff_pose = torch.abs(input[:, :11] - target[:, :11])
        for ind in range(11):
            if ind in [0, 4, 8]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 0]
            elif ind in [1, 5, 9]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 1]
            elif ind in [2, 6, 10]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 2]
            else:
                weights[:, ind] = param_diff_pose[:, ind] * offset_norm

        ## This is the optimizest version
        # for shape_exp
        magic_number = 0.00057339936  # scale
        param_diff_shape_exp = torch.abs(input[:, 12:] - target[:, 12:])
        # weights[:, 12:] = magic_number * param_diff_shape_exp * self.w_norm
        w = torch.cat((w_shp_base, w_exp_base), dim=1)
        w_norm = torch.norm(w, dim=0)
        # print('here')
        weights[:, 12:] = magic_number * param_diff_shape_exp * w_norm

        eps = 1e-6
        weights[:, :11] += eps
        weights[:, 12:] += eps

        # normalize the weights
        maxes, _ = weights.max(dim=1)
        maxes = maxes.view(-1, 1)
        weights /= maxes

        # zero the z
        weights[:, 11] = 0

        return weights

    def forward(self, input, target, weights_scale=10):
        if self.opt_style == 'resample':
            weights = self._calc_weights_resample(input, target)
            loss = weights * (input - target) ** 2
            return loss.mean()
        else:
            raise Exception(f'Unknown opt style: {self.opt_style}')


if __name__ == '__main__':
    pass
