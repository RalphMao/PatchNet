from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolSelect(nn.Module):
    def __init__(self, select_size=1, **pool_kwargs):
        super(MaxPoolSelect, self).__init__()
        self.select_size = select_size
        self.pool_kwargs = pool_kwargs

    def forward(self, score, offset):
        score_shape = score.shape
        score_shape[1] *= self.select_size
        assert (score_shape == offset.shape).all()
        score_pool, indices = F.max_pool2d(score, return_indices=True, **self.pool_kwargs)
        pool_shape = score_pool.shape
        pool_shape[1] *= self.select_size
        i_offset = self._flatten(indices, 2).unsqueeze(2).repeat(self.select_size)
        i_offset = self._flatten(i_offset, 2, 3)
        offset_select = torch.gather(self._flatten(offset, 2), i_offset).view(pool_shape)
        return score_pool, offset_select

    @staticmethod
    def _flatten(data, start_dim=0, end_dim=-1):
        if torch.__version__.startswith('1'):
            return torch.flatten(data, start_dim, end_dim)
        else:
            shape = tuple(data.shape)
            end_dim = len(shape) + end_dim if end_dim < 0 else end_dim
            new_shape = shape[:start_dim] + (-1, ) + shape[end_dim+1:]
            return data.view(*new_shape)

class MaxPoolSoftSelect(nn.Module):
    def __init__(self, ks, stride):
        super(MaxPoolSoftSelect, self).__init__()
        self.ks = ks
        self.stride = stride
        self.offset_addi = nn.Parameter(self._get_offset_bias(ks)[..., None, None])
        self.alpha = nn.Parameter(torch.ones(1))

    @staticmethod
    def _get_offset_bias(ks):
        offset = torch.zeros(4, ks, ks)
        offset[:2, :, :] = torch.arange(ks, dtype=torch.float)[None, :, None] - (ks - 1.0) / 2
        offset[2:, :, :] = torch.arange(ks, dtype=torch.float)[None, None, :] - (ks - 1.0) / 2
        return offset.view(4, -1)

    def forward(self, score, offset):
        n, c, H, W = score.shape
        ks, stride = self.ks, self.stride
        h = int(np.ceil((H - ks + 1.0) / stride))
        w = int(np.ceil((W - ks + 1.0) / stride))

        score_fold = F.unfold(score, ks, stride=stride)
        score_fold = score_fold.view(n, c, 1, ks*ks, h, w)
        score_pool, _ = torch.max(score_fold, dim=3)
        score_pool = score_pool.squeeze(2)

        offset_fold = F.unfold(offset, ks, stride=stride)
        offset_fold = offset_fold.view(n, c, 4, ks*ks, h, w)
        score_normalize = F.softmax(score_fold.detach() * self.alpha, dim=3)
        offset_pool = torch.sum(score_normalize * (offset_fold + self.offset_addi), dim=3)
        offset_pool = offset_pool.view(n, c*4, h, w)
        return score_pool, offset_pool
    
