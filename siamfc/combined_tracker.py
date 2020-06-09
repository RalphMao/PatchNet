from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple

from .siamfc import TrackerSiamFC
from . import ops

class TrackerCombined(TrackerSiamFC):
    def __init__(self, net_path, patchnet, factor=0.1, **kwargs):
        self.factor = factor
        self.patchnet = patchnet
        super().__init__(net_path, **kwargs)

    @torch.no_grad()
    def init(self, img, box):
        super().init(img, box)
        self.img_ckpt = img
        self.box_ckpt = box

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        siam_responses = responses.squeeze(1).cpu().numpy()

        # PatchNet responses
        imgs = np.stack((self.img_ckpt, img))
        imgs = imgs[:,:,:,::-1]
        box = np.array([
            self.center[1] - (self.target_sz[1] - 1) / 2,
            self.center[0] - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        corr_boxes = np.stack((self.box_ckpt, box))
        corr_boxes[0,:2] -= 1
        corr_boxes[:,2:] += corr_boxes[:, :2]
        corr_boxes = corr_boxes[:,None]

        images = self.patchnet.preprocess_image(imgs)
        corr_boxes = torch.from_numpy(corr_boxes).float().cuda()
        base_boxes = corr_boxes[0]
        target_boxes = corr_boxes[1]
        x, z = self.patchnet._get_target_search_features(images, base_boxes, target_boxes, mode="TEST")
        score_maps, offset_maps = self.patchnet.forward(x, z)
        score_maps = score_maps.cpu().numpy()[0,0]

        # upsample responses and penalize scale changes
        presp = cv2.resize(
            score_maps, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
        siam_responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in siam_responses]) 
        responses = siam_responses + presp * self.factor
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box, np.ones(1)
    

