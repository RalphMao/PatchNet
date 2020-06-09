from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import tile2d

class FourierFeature(nn.Module):
    def __init__(self, N, K, tiled=False):
        super(FourierFeature, self).__init__()
        self.n = N
        self.k = K
        k_fft = K // 2 + 1
        self.freq_coeff = nn.Parameter(torch.ones(K, k_fft, 2))
        self.tiled = tiled

    def forward(self, x, z):
        self.w = self.get_feature_weights(z)
        return self.forward_with_weights(x, self.w)

    def forward_with_weights(self, data, weight):
        b, c, h, w = data.shape
        data = data.view(-1, b * c, h, w)
        resp = F.conv2d(data, weight, stride=2, groups=b)
        normalize_term = F.conv2d(data * data, torch.ones_like(weight), stride=2, groups=b)
        normalize_term += torch.std(normalize_term) / 10 + 1e-9
        normalize_term = 1e-2 / torch.sqrt(normalize_term).detach()
        # normalize_term = 1e-5
        resp = resp * normalize_term
        n, c, h, w = resp.shape
        assert c % b == 0 and n == 1
        return resp.view(b, c // b, h, w)

    def get_feature_weights(self, patch):
        n, k = self.n, self.k
        assert patch.ndimension() == 4
        batch_size = patch.shape[0]
        c = patch.shape[1]
        assert patch.shape[2] == n * self.k and patch.shape[2] ==patch.shape[3]
        w1 = patch.view(batch_size, c, n, k, n, k).permute(0,2,4,1,3,5).contiguous()
        '''
        w1_flatten = w1.view(w1.shape[0], w1.shape[1] * w1.shape[2], -1)
        normalize_term = torch.sqrt(torch.sum(w1_flatten ** 2, dim=2))
        normalize_term = torch.mean(normalize_term, dim=1) + 1e-6
        w1 = w1 / normalize_term[:, None, None, None, None, None]
        '''
        w1 = w1.view(batch_size, n * n, c, k, k)
        if self.tiled:
            w1 = tile2d(w1, 1)
        w1 = w1.view(batch_size * n * n, c, k, k)
        w1_f = torch.rfft(w1, signal_ndim=2, normalized=True, onesided=True) * self.freq_coeff
        w1_new = torch.irfft(w1_f, signal_ndim=2, normalized=True, onesided=True, signal_sizes=w1.shape[-2:])

        return w1_new

class FeatureContrast(nn.Module):
    def __init__(self, n, kk):
        super(FeatureContrast, self).__init__()
        self.n = n
        self.kk = kk
        self.inds = nn.Parameter(self._create_inds(), requires_grad=False)
        self.aggregate_inds = nn.Parameter(self._create_aggregate_weights(n, kk), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(1))
    
    def _create_inds(self):
        n = self.n
        channel_inds = torch.arange(n * n).long().view(n, n)
        channel_inds += 1 + torch.arange(n).long()[:, None] * 2
        channel_inds += n+2
        return channel_inds.view(-1)
    
    def forward(self, x):
        valid_x = torch.index_select(x, dim=1, index=self.inds)
        xexp = torch.exp(x.detach() * self.weight)
        valid_xexp = torch.index_select(xexp, dim=1, index=self.inds)
        imp = valid_xexp / F.conv2d(xexp, self.aggregate_inds)
        return valid_x * imp

    def _create_aggregate_weights(self, n, kk):
        n_in = n + 2 * kk
        n_out = n
        weight = torch.zeros(n_out, n_out, n_in, n_in)
        for i in range(n_out):
            for j in range(n_out):
                weight[i, j, i:i + 2 * kk + 1, j:j + 2 * kk + 1] = 1
        return weight.view(n_out * n_out, n_in * n_in, 1, 1)
