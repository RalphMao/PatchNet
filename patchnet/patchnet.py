from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_feature import FourierFeature, FeatureContrast
from .pool_select import MaxPoolSoftSelect
from .utils import find_peaks, cuttop, tile2d, conv2gconv

__all__ = ["PatchNet", "PatchNetG", "PatchAggregate"]

def crop_pool(bottom, rois, pool_size, feat_stride=1.0):
    rois = rois.detach()

    x1 = rois[:, 1::4] / float(feat_stride)
    y1 = rois[:, 2::4] / float(feat_stride)
    x2 = rois[:, 3::4] / float(feat_stride)
    y2 = rois[:, 4::4] / float(feat_stride)

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    theta = rois.data.new(rois.size(0), 2, 3).zero_()
    theta[:, 0:1, 0] = (x2 - x1) / (width - 1)
    theta[:, 0:1 ,2] = (x1 + x2 - width + 1) / (width - 1)
    theta[:, 1:2, 1] = (y2 - y1) / (height - 1)
    theta[:, 1:2, 2] = (y1 + y2 - height + 1) / (height - 1)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pool_size, pool_size)))
    crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)

    return crops


class ROITrackBase(nn.Module):
    def __init__(self):
        super(ROITrackBase, self).__init__()
        self._search_size = None
        self._feat_stride = None
        self._resp_stride = None
        self._pool_size = None
        self._pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)
        self._pixel_means_th = torch.from_numpy(self._pixel_means).cuda()

    @property
    def _search_ratio(self):
        return float(self._search_size) / self._pool_size

    def _get_features(self, feature_maps, boxes, search=False):
        num_box = boxes.shape[0]
        sizes = boxes[:, 2:] - boxes[:, :2]
        centers = (boxes[:, 2:] + boxes[:, :2]) / 2
        rois = torch.zeros(num_box, 5).cuda()
        if search:
            pool_size = self._search_size
        else:
            pool_size = self._pool_size
        rois[:, 1:3] = centers - pool_size / 2.0
        rois[:, 3:5] = centers + 1 + pool_size / 2.0
        x = crop_pool(feature_maps, rois, pool_size, self._feat_stride)
        raise NotImplementedError
        return x
       
    def _get_target_search_features(self, feature_maps, exemplar_boxes, target_boxes, mode='TRAIN'):
        '''
        if mode=TRAIN, only center of target_boxes is used
        ''' 
        num_box = exemplar_boxes.shape[0]
        exemplar_sizes = exemplar_boxes[:, 2:] - exemplar_boxes[:, :2]
        if mode == 'TRAIN':
            search_sizes = exemplar_sizes
        else:
            search_sizes = target_boxes[:, 2:] - target_boxes[:, :2]
        search_centers = (target_boxes[:,:2] + target_boxes[:, 2:]) / 2
        search_boxes = exemplar_boxes.clone()
        search_boxes[:, :2] = search_centers - search_sizes * self._search_ratio / 2
        search_boxes[:, 2:] = search_centers + search_sizes * self._search_ratio / 2

        exemplar_rois = torch.zeros(num_box, 5).cuda()
        search_rois = torch.zeros(num_box, 5).cuda()
        exemplar_rois[:, 1:] = exemplar_boxes
        exemplar_rois[:, 3:] += 1
        search_rois[:, 1:] = search_boxes
        search_rois[:, 3:] += 1

        z = crop_pool(feature_maps[:1], exemplar_rois, self._pool_size, self._feat_stride)
        x = crop_pool(feature_maps[1:], search_rois, self._search_size, self._feat_stride)
        return x, z

    @staticmethod
    def _vis_kernel(kernel):
        if kernel.ndimension() == 3:
            kernel = kernel.permute(1,2,0).cpu().numpy()
        else:
            kernel = kernel.cpu().numpy()
        kernel = kernel - kernel.min()
        kernel = np.floor(kernel / kernel.max() * 255).astype(np.uint8)
        cv2.imshow("win2", kernel)
        cv2.waitKey()

    def _estimate_peak(self, responses):
        responses = responses.detach()
        X, N, H, W = responses.shape
        # Naive version
        responses_f = responses.view(X * N, -1)
        if self.scoretype == 'plain':
            peak_scores, locs = torch.max(responses_f, dim=-1)
        elif self.scoretype == 'peak':
            peak_scores, locs = find_peaks(responses.view(X * N, H, W))
        elif self.scoretype == 'temporal':
            w = self.feature_layer.w
            weight_coeff = torch.sum(w * w)
            weight_coeff = torch.sqrt(weight_coeff / w.shape[0]) / 40
            responses_f /= weight_coeff
            peak_scores, locs = torch.max(responses_f, dim=-1)
        elif self.scoretype == 'combined':
            peak_scores_s, locs = find_peaks(responses.view(X * N, H, W))
            w = self.feature_layer.w
            weight_coeff = torch.sum(w * w)
            weight_coeff = torch.sqrt(weight_coeff / w.shape[0]) / 40
            responses_f /= weight_coeff
            peak_scores_t, locs_ = torch.max(responses_f, dim=-1)
            peak_scores_t = cuttop(peak_scores_t, 1)
            peak_scores = peak_scores_s * peak_scores_t
            
        # assert (locs_ == locs).all() or (peak_scores == 0).all()

        x, y = locs % W, locs / W
        x, y = x.float(), y.float()
        x, y = x - float(W - 1) / 2, y - float(H - 1) / 2
        max_locs = torch.stack((x, y), dim=-1)
        return peak_scores.detach(), max_locs, locs

    def _translate_boxes(self, base_boxes, max_locs):
        box_sizes = base_boxes[:, 2:] - base_boxes[:, :2]
        locs_offset = max_locs * self._resp_stride * box_sizes / self._pool_size
        boxes = base_boxes.copy()
        boxes[:, :2] += locs_offset
        boxes[:, 2:] += locs_offset
        return boxes

    def _add_offsets(self, base_boxes, offsets):
        box_sizes = base_boxes[:, 2:] - base_boxes[:, :2]
        box_sizes = np.tile(box_sizes, (1, 2))
        return base_boxes + offsets * self._resp_stride * box_sizes / self._pool_size
        
class PatchAggregate(ROITrackBase):
    def __init__(self, K=8, N=8, resp_size=7, backtrace=False, scoretype="peak", border_normalize=False):
        super(PatchAggregate, self).__init__()
        self.k = K # Feature kernel size
        self.n = N # kernel number per dimension
        self._feat_stride = 1
        self._resp_stride = N
        self._pool_size = K * N
        self.set_resp_size(resp_size)
        self.backtrace_flag = backtrace
        self.scoretype = scoretype
        self.border_normalize = border_normalize

    def set_resp_size(self, resp_size):
        self._resp_size = resp_size
        self._search_size = 63 + resp_size * 8
    
    def create_architecture(self):
        n, k = self.n, self.k
        k2 = k // 2 + 1
        self.w2 = nn.Parameter(self.create_structural_weights(n, k2))
        n2 = n // 2
        self.w3 = nn.Parameter(self.create_structural_weights(n2, k2))
        n3 = n2 // 2
        self.w4 = nn.Parameter(self.create_structural_weights(n3, k2))
        self.offset_addi = nn.Parameter(self._get_offset_bias(2)[..., None, None])

    def create_feature_weights(self, patch):
        n, k = self.n, self.k
        assert patch.ndimension() == 4
        batch_size = patch.shape[0]
        c = patch.shape[1]
        assert patch.shape[2] == n * self.k and patch.shape[2] ==patch.shape[3]
        patch_mean = patch.view(batch_size, c, -1).mean(dim=2)
        patch = patch - patch_mean[..., None, None]
        w1 = patch.view(batch_size, c, n, k, n, k).permute(0,2,4,1,3,5).contiguous()
        w1_flatten = w1.view(w1.shape[0], w1.shape[1] * w1.shape[2], -1)
        normalize_term = torch.sqrt(torch.sum(w1_flatten ** 2, dim=2))
        normalize_term = torch.mean(normalize_term, dim=1) + 1e-6
        w1 = w1 / normalize_term[:, None, None, None, None, None]
        w1 = w1.view(batch_size * n * n, c, k, k)
        return w1, patch_mean 

    @staticmethod
    def create_structural_weights(n, k, factor=2):
        assert n % factor == 0
        n_out = n // factor
        weight = torch.zeros(n_out, n_out, n, n, k, k)
        coeff = 1.0 / factor / factor
        for i in range(n_out):
            for j in range(n_out):
                for x in range(factor):
                    for y in range(factor):
                        weight[i,j,i*factor+x, j*factor+y, (k-1)*x, (k-1)*y] = coeff
        weight = weight.view(n_out * n_out, n * n, k, k)
        return weight

    @staticmethod
    def create_offset_base(factor):
        assert factor == 2
        # Method 1 by Least-Square Estimation
        a = np.array([[1,0],[0.5,0.5],[0.5,0.5],[0,1]])
        base_weight = np.dot(np.linalg.inv(np.dot(a.T, a)), a.T).reshape((2, factor, 2))
        return torch.from_numpy(base_weight).permute(1,0,2) / (factor ** 2)

    @staticmethod
    def _get_offset_bias(ks):
        offset = torch.zeros(4, ks, ks)
        offset[:2, :, :] = torch.arange(ks, dtype=torch.float)[None, :, None] - (ks - 1.0) / 2
        offset[2:, :, :] = torch.arange(ks, dtype=torch.float)[None, None, :] - (ks - 1.0) / 2
        return offset.view(4, -1)
        
        
    def create_offset_weights(self, n, k, factor=2):
        '''
        Offset is a 4-dim array of shape N, (C, 4), H, W
        Each offset element corresponds to (ymin_shift, ymax_shift, xmin_shift, xmax_shift)
        '''
        assert n % factor == 0
        n_out = n // factor
        weight = torch.zeros(n_out, n_out, 4, n, n, 4, k, k)
        base_weight = self.create_offset_base(factor)
        for i in range(n_out):
            for j in range(n_out):
                for x in range(factor):
                    for y in range(factor):
                        weight[i, j, :2, i*factor+x, j*factor+y, :2, (k-1)*x, (k-1)*y] = base_weight[x]
                        weight[i, j, 2:, i*factor+x, j*factor+y, 2:, (k-1)*x, (k-1)*y] = base_weight[y]
        weight = weight.view(n_out * n_out * 4, n * n * 4, k, k)
        return weight

    def forward(self, x, z):
        self.w1, self.patch_mean = self.create_feature_weights(z)
        self.patch_mean = self.patch_mean.detach()
        x = x - self.patch_mean[..., None, None]

        feature_maps = self.feature_corr(x, self.w1)
        if self.backtrace_flag:
            feature_maps.retain_grad()
            self.feature_maps = feature_maps
        score_maps, offset_maps = self.aggregate_score(feature_maps)
        self.responses = score_maps
        return score_maps, offset_maps


    @staticmethod
    def _select_offsets(offset_maps, max_idxs):
        offsets = np.zeros((offset_maps.shape[0], offset_maps.shape[1]))
        offset_maps = offset_maps.reshape((offset_maps.shape[0], offset_maps.shape[1], -1))
        for i, j in enumerate(max_idxs):
            offsets[i] = offset_maps[i, [2,0,3,1], j]
        return offsets

    def preprocess_image_cuda(self, images):
        images = torch.from_numpy(images.copy()).cuda().float()
        images -= self._pixel_means_th
        images = images.permute(0,3,1,2).contiguous()
        return images

    def init(self, images, corr_boxes):
        images = self.preprocess_image_cuda(images)
        corr_boxes = torch.from_numpy(corr_boxes).float().cuda()
        assert images.ndimension() == 4 and corr_boxes.ndimension() == 3
        assert images.shape[0] == corr_boxes.shape[0]
        assert images.shape[0] == 1
        z = self._get_features(images, corr_boxes[0])
        self.w1, self.patch_mean = self.create_feature_weights(z)
        
    def track(self, images, corr_boxes):
        '''
        Input: images in ndarray
        Output: scores in ndarray
        '''
        self.eval()
        if corr_boxes.size == 0:
            return np.zeros(0), np.zeros((0, 4))
        images = self.preprocess_image_cuda(images)
        corr_boxes = torch.from_numpy(corr_boxes).float().cuda()
        assert images.ndimension() == 4 and corr_boxes.ndimension() == 3
        assert images.shape[0] == corr_boxes.shape[0]

        images.requires_grad = self.backtrace_flag
        if images.shape[0] == 1:
            base_boxes = corr_boxes[0]
            target_boxes = corr_boxes[0]
            x = self._get_features(images, base_boxes, search=True)
            z = None
            raise NotImplementedError
        elif images.shape[0] == 2:
            base_boxes = corr_boxes[0]
            target_boxes = corr_boxes[1]
            x, z = self._get_target_search_features(images, base_boxes, target_boxes, mode="TEST")
            num_template = 1
        else:
            # Multi-template tracking
            target_boxes = corr_boxes[-1]
            num_template = len(corr_boxes) - 1

            xs = []
            zs = []
            for idx in range(num_template):
                x_tmp, z_tmp = self._get_target_search_features(images[[idx, -1]], corr_boxes[idx], target_boxes, mode="TEST")
                xs.append(x_tmp)
                zs.append(z_tmp)
            x = torch.cat(xs)
            z = torch.cat(zs)

        if self.backtrace_flag:
            x.retain_grad()

        score_maps, offset_maps = self.forward(x, z)
        if num_template > 1:
            score_maps = torch.mean(score_maps, dim=0)[None]
            # offset_maps = torch.mean(offset_maps, dim=0)[None]
            offset_maps = offset_maps[-1:]
        self.responses = score_maps

        peak_scores, max_locs, max_idxs = self._estimate_peak(score_maps)
        if self.backtrace_flag:
            loss = 0
            score_maps_flatten = score_maps.view(score_maps.shape[0], -1)
            for idx, max_idx in enumerate(max_idxs):
                loss += score_maps_flatten[idx, max_idx]
            loss.backward()
            mask = torch.abs(images.grad) > 1e-9
            self.mask = mask[-1,0,...,None].cpu().numpy().astype(np.float32)
            xmask = torch.abs(x.grad) > 1e-9
            self.xmask = xmask.cpu().numpy()
            fmask = torch.abs(self.feature_maps.grad) > 1e-9
            self.fmask = fmask.cpu().numpy()


        peak_scores = peak_scores.cpu().numpy()
        target_boxes = target_boxes.cpu().numpy()
        max_locs = max_locs.cpu().numpy()
        max_idxs = max_idxs.cpu().numpy()
        offset_maps = offset_maps.detach().cpu().numpy()

        shifted_boxes = self._translate_boxes(target_boxes, max_locs)
        offsets = self._select_offsets(offset_maps, max_idxs)
        new_boxes = self._add_offsets(shifted_boxes, offsets)

        return peak_scores, new_boxes

    def feature_corr(self, data, w1):
        b, c, h, w = data.shape
        data = data.view(-1, b * c, h, w)
        resp = F.conv2d(data, w1, groups=b)
        normalize_term = F.conv2d(data * data, torch.ones_like(w1), groups=b)
        normalize_term += torch.std(normalize_term) / 10 + 1e-9
        normalize_term = torch.sqrt(normalize_term).detach()
        resp = resp / normalize_term
        n, c, h, w = resp.shape
        assert c % b == 0 and n == 1
        return resp.view(b, c // b, h, w)

    def aggregate_score(self, feature_scores):
        in2 = F.max_pool2d(feature_scores, 2, padding=0, stride=2)
        # in2 = feature_scores
        out2 = F.conv2d(in2, self.w2)
        out2p = F.max_pool2d(out2, 2, stride=2)
        out3 = F.conv2d(out2p, self.w3)
        out3p = F.max_pool2d(out3, 2, stride=2)
        out4 = F.conv2d(out3p, self.w4)
        reg4 = torch.zeros_like(out4).repeat(1,4,1,1)

        return out4, reg4

class PatchNet(PatchAggregate):
    def _create_feature_layer(self, initialize_weight=True):
        self.feature_layer = FourierFeature(self.n, self.k)

    def create_architecture(self, initialize_weight=True):
        self._create_feature_layer(initialize_weight=initialize_weight)
        self._create_aggregate_layer(initialize_weight=initialize_weight)
        
    def _create_aggregate_layer(self, initialize_weight=True):
        n, k = self.n, self.k
        k2 = k // 2 + 1
        n2 = n // 2
        n3 = n2 // 2
        self.conv2 = nn.Conv2d(in_channels=n*n, out_channels=n2 * n2,
                               kernel_size=k2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=n2*n2, out_channels=n3 * n3,
                               kernel_size=k2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=n3*n3, out_channels=1,
                               kernel_size=k2, bias=False)

        self.conv3reg = nn.Conv2d(in_channels=4*n2*n2, out_channels=4*n3 * n3,
                               kernel_size=k2, bias=False)
        self.conv4reg = nn.Conv2d(in_channels=4*n3*n3, out_channels=4,
                               kernel_size=k2, bias=False)
        
        self.pool2 = MaxPoolSoftSelect(ks=2, stride=2)
        self.pool3 = MaxPoolSoftSelect(ks=2, stride=2)

        if initialize_weight:
            self.conv2.load_state_dict({'weight':self.create_structural_weights(n, k2)})
            self.conv3.load_state_dict({'weight':self.create_structural_weights(n2, k2)})
            self.conv4.load_state_dict({'weight':self.create_structural_weights(n3, k2)})
            self.conv3reg.load_state_dict({'weight':self.create_offset_weights(n2, k2)})
            self.conv4reg.load_state_dict({'weight':self.create_offset_weights(n3, k2)})

    def forward(self, x, z):
        feature_maps = self.feature_layer(x, z)
        score_maps, offset_maps = self.aggregate_score(feature_maps)
        return score_maps, offset_maps

    def aggregate_score(self, feature_scores):
        out2 = self.conv2(feature_scores)                
        reg2 = torch.zeros_like(out2).repeat(1,4,1,1)
        out2p, reg2p = self.pool2(out2, reg2)
        out3 = self.conv3(out2p)
        reg3 = self.conv3reg(reg2p)
        out3p, reg3p = self.pool3(out3, reg3)
        out4 = self.conv4(out3p)
        reg4 = self.conv4reg(reg3p)
        return out4, reg4

class PatchNetG(PatchNet):
    def _create_feature_layer(self, initialize_weight=True):
        self.feature_layer = FourierFeature(self.n, self.k, tiled=True)

    def _create_aggregate_layer(self, initialize_weight=True):
        n, k = self.n, self.k
        k2 = k // 2 + 1
        n2 = n // 2
        n3 = n2 // 2
        self.conv2 = nn.Conv2d(in_channels=n*n, out_channels=n2 * n2,
                               kernel_size=k2, bias=False, groups=n2 * n2)
        self.conv3 = nn.Conv2d(in_channels=n2*n2, out_channels=n3 * n3,
                               kernel_size=k2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=n3*n3, out_channels=1,
                               kernel_size=k2, bias=False)

        self.conv3reg = nn.Conv2d(in_channels=4*n2*n2, out_channels=4*n3 * n3,
                               kernel_size=k2, bias=False)
        self.conv4reg = nn.Conv2d(in_channels=4*n3*n3, out_channels=4,
                               kernel_size=k2, bias=False)
        
        self.pool2 = MaxPoolSoftSelect(ks=2, stride=2)
        self.pool3 = MaxPoolSoftSelect(ks=2, stride=2)

        if initialize_weight:
            weight = tile2d(tile2d(self.create_structural_weights(n, k2), 1), 0)
            self.conv2.load_state_dict({'weight':conv2gconv(weight, n2*n2)})
            self.conv3.load_state_dict({'weight':self.create_structural_weights(n2, k2)})
            self.conv4.load_state_dict({'weight':self.create_structural_weights(n3, k2)})
            self.conv3reg.load_state_dict({'weight':self.create_offset_weights(n2, k2)})
            self.conv4reg.load_state_dict({'weight':self.create_offset_weights(n3, k2)})

    def aggregate_score(self, feature_scores):
        out2 = self.conv2(feature_scores)                
        out2 = tile2d(out2, 1)
        reg2 = torch.zeros_like(out2).repeat(1,4,1,1)
        out2p, reg2p = self.pool2(out2, reg2)
        out3 = self.conv3(out2p)
        reg3 = self.conv3reg(reg2p)
        out3p, reg3p = self.pool3(out3, reg3)
        out4 = self.conv4(out3p)
        reg4 = self.conv4reg(reg3p)
        return out4, reg4

