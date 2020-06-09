from __future__ import absolute_import, division, print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from got10k.trackers import Tracker
from patchnet import PatchNet

from .optim import AdamSparse
from .losses import BalancedLoss, CenterLoss
from .datasets import Pair
from .ops import show_image
from .metric import iou_box

__all__ = ['TrainerPatchNet']

class TrainerPatchNet(Tracker):
    def __init__(self):
        super(TrackerPatchNet, self).__init__(self.__class__.__name__, True)
        self.net = PatchNet()
        self.net.create_architecture()
        self.net.cuda()
        self.criterion = CenterLoss()
        self.offset_criterion = nn.SmoothL1Loss()
        self.criterion.cuda()
        # self.net.feature_layer.freq_coeff.requires_grad = False
        self.optimizer = AdamSparse(self.net.parameters(),
                lr=1e-3,
                weight_decay=5e-4)
        self.pixel_means = torch.from_numpy(self.net._pixel_means).cuda().float()

    def train_over(self, seqs, args, val_seqs=None, save_dir='pretrained'):
        dataset = Pair(
            seqs=seqs,
            max_distance=12)
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        flog = open(os.path.join(save_dir, 'patchnet.log'), 'w')
        if val_seqs is not None:
            val_dataset = Pair(seqs=val_seqs,
                               max_distance=12,
                               pairs_per_seq=2,
                               fixed=True)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True)

            val_loss, val_metric = self.run_epoch(val_dataloader, train_flag=False)
            msg = 'Epoch: {} Val Loss: {:.5f}, Accuracy {:.5f}'.format(0, val_loss, val_metric)
            print(msg)
            print(msg, file=flog, flush=True)

        for epoch in range(args.epoch_num):

            print('Epoch: {} Training in Progress'.format(epoch))
            train_loss, train_metric = self.run_epoch(dataloader, train_flag=True)
            msg = 'Epoch: {} Train Loss: {:.5f}, Accuracy {:.5f}'.format(epoch, train_loss, train_metric)
            print(msg)
            print(msg, file=flog, flush=True)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'patchnet_%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

            val_loss, val_metric = self.run_epoch(val_dataloader, train_flag=False)
            msg = 'Epoch: {} Val Loss: {:.5f}, Accuracy {:.5f}'.format(epoch, val_loss, val_metric)
            print(msg)
            print(msg, file=flog, flush=True)

    def run_epoch(self, dataloader, train_flag):
        total_loss = 0
        tps = 0.0
        for it, batch in enumerate(dataloader):
            loss, tp = self.train_step(batch, train_flag=train_flag)
            total_loss += loss
            tps += tp
            if tp < 0.5:
                show_image(batch[0][0].cpu().numpy(), batch[2][0].cpu().numpy(), fig_n=it*2)
                show_image(batch[1][0].cpu().numpy(), batch[3][0].cpu().numpy(), fig_n=it*2+1)
        avg_loss = total_loss / len(dataloader)
        avg_tps = tps / len(dataloader)
        return avg_loss, avg_tps

    def preprocess(self, batch):
        images = torch.cat(batch[:2]).cuda().float()
        images = (images - self.pixel_means).permute(0,3,1,2)
        base_boxes = batch[2].float().cuda()
        base_boxes[:, :2] -= 1
        base_boxes[:, 2:] += base_boxes[:, :2]
        target_boxes = batch[3].float().cuda()
        target_boxes[:, :2] -= 1
        target_boxes[:, 2:] += target_boxes[:, :2]
        try:
            assert base_boxes[0,2] <= images.shape[3], "%f vs %d"%(base_boxes[0,2], images.shape[3])
            assert base_boxes[0,3] <= images.shape[2], "%f vs %d"%(base_boxes[0,3], images.shape[2])
        except:
            pass
        x, z = self.net._get_target_search_features(images, base_boxes, target_boxes, mode="TRAIN")
        return x, z

    def preprocess2(self, batch):
        images = torch.cat(batch[:2]).cuda().float()
        images = (images - self.pixel_means).permute(0,3,1,2)
        base_boxes = batch[2].float()
        base_boxes[:, :2] -= 1
        base_boxes[:, 2:] += base_boxes[:, :2]
        target_boxes = batch[3].float()
        target_boxes[:, :2] -= 1
        target_boxes[:, 2:] += target_boxes[:, :2]
        base_center = (base_boxes[:, :2] + base_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        box_sizes = base_boxes[:, 2:] - base_boxes[:, :2]


        real_to_affine_coeff = self.net._pool_size / self.net._resp_stride / box_sizes
        affine_to_real_coeff = 1.0 / real_to_affine_coeff

        center_affine = (target_center - base_center) * real_to_affine_coeff
        approx_center_affine = torch.round(center_affine)

        # When center shift is going out of max shift range
        max_shift = (self.net._resp_size - 1) // 2
        in_range_shift_affine = torch.sign(approx_center_affine) * (approx_center_affine.abs() // (max_shift + 1)) * (max_shift + 1)
        in_range_approx_center_affine = approx_center_affine - in_range_shift_affine
        in_range_shift = in_range_shift_affine * affine_to_real_coeff
        base_boxes_shift = base_boxes + in_range_shift.repeat(1,2)


        target_boxes_affine = (target_boxes - base_boxes) * self.net._pool_size / self.net._resp_stride / box_sizes.repeat(1,2)
        offsets = target_boxes_affine - approx_center_affine.repeat(1,2)
        base_boxes = base_boxes.cuda()
        target_boxes = target_boxes.cuda()
        offsets = offsets.cuda()
        center_affine = center_affine.cuda()
        x, z = self.net._get_target_search_features(images, base_boxes, base_boxes_shift, mode="TEST")
        return x, z, in_range_approx_center_affine.long(), offsets, base_boxes_shift, target_boxes


    def train_step(self, batch, train_flag=True):
        self.net.train(train_flag)
        with torch.set_grad_enabled(train_flag):
            x, z, center_rel_gt, offset_gt, base_boxes, target_boxes = self.preprocess2(batch)
            score_map, offset_map = self.net(x, z)
            k = (score_map.shape[-1] - 1) // 2
            center_abs_gt = center_rel_gt + k 
            center_loss = self.criterion(score_map, center_abs_gt[0]) / 49 * 4
            offset_pred = offset_map[:, [2,0,3,1], center_abs_gt[0, 1], center_abs_gt[0, 0]]
            offset_loss = self.offset_criterion(offset_pred*10, offset_gt*10)
            loss = offset_loss + center_loss

            peak_scores, max_locs, max_idxs = self.net._estimate_peak(score_map)
            target_boxes = target_boxes.cpu().numpy()       
            base_boxes = base_boxes.cpu().numpy()       
            max_locs = max_locs.cpu().numpy()               
            max_idxs = max_idxs.cpu().numpy()               
            offset_map = offset_map.detach().cpu().numpy()
            shifted_boxes = self.net._translate_boxes(base_boxes, max_locs)
            offsets = self.net._select_offsets(offset_map, max_idxs)        
            new_boxes = self.net._add_offsets(shifted_boxes, offsets)        

            shifted_boxes_gt = self.net._translate_boxes(base_boxes, center_rel_gt.cpu().numpy())
            target_boxes_gt = self.net._add_offsets(shifted_boxes_gt, offset_gt.cpu().numpy())
            # For debug: assert (target_boxes - target_boxes_gt).sum() < 1e-1

            iou = iou_box(new_boxes[0], target_boxes[0])
            if train_flag:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss.item(), iou

    def _create_labels(self, resp):
        size = resp.size()
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = 0.5
        r_neg = 0
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(resp.device).float()
        
        return self.labels


