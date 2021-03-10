from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data as data
import torch
from collections import defaultdict
import numpy as np
import cv2
import argparse
import glob
import sys
import os

sys.path.insert(0, './')
from patchnet.patchnet import PatchNet, PatchAggregate


def clip_box(bboxes, frame_size):
    if len(bboxes) > 0:
        bboxes[:, :2] = np.maximum(bboxes[:, :2], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], frame_size[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], frame_size[1])


def draw_detection(im, bboxes, scores, cls_inds, labels, color=(255, 0, 0), width=1.0, extra=None):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    bboxes = bboxes.astype('i')
    if extra is not None:
        assert len(extra) == len(
            bboxes), "Number of extra texts does not match!"
    else:
        extra = [''] * len(bboxes)

    for i, box in enumerate(bboxes):

        thick = int(max(h, w) / 600 * width)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)
        mess = extra[i]
        if cls_inds is not None and labels is not None:
            cls_indx = cls_inds[i]
            mess += labels[cls_indx] + ' '
        if scores is not None:
            mess += '%.3f' % scores[i]
        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                    0, 1e-3 * h, color, thick // 2)

    return imgcv


def filter_byconf(scores, conf, *args):
    if hasattr(scores, 'keys'):
        for frame in scores:
            arg_elements = [arg[frame] for arg in args]
            results = filter_byconf(scores[frame], conf, *arg_elements)
            for arg, result in zip(args, results[1:]):
                arg[frame] = result
            scores[frame] = results[0]
        return (scores, ) + args
    else:
        keep = tuple(filter(lambda x:scores[x] >= conf, np.arange(len(scores))))
        keep = np.array(keep)
        args = tuple(map(lambda x:x[keep], args))
        return (scores[keep], ) + args


def read_det_file(filename):
    bboxes = defaultdict(list)
    scores = defaultdict(list)
    cls_inds = defaultdict(list)
    for line in open(filename):
        frame_id, cls, score, xmin, ymin, xmax, ymax = [float(field) for field in line.split()]
        frame_id = int(frame_id)
        bboxes[frame_id].append((xmin, ymin, xmax, ymax))
        scores[frame_id].append(score)
        cls_inds[frame_id].append(int(cls))
    for frame_id in bboxes:
        bboxes[frame_id] = np.array(bboxes[frame_id])
        scores[frame_id] = np.array(scores[frame_id])
        cls_inds[frame_id] = np.array(cls_inds[frame_id])
    return bboxes, scores, cls_inds


class MultiTracker(object):
    def __init__(self, network, frame, boxes):
        self.network = network
        self.initial_frame = frame
        self.initial_boxes = boxes
        self.previous_boxes = boxes

    def update(self, img):
        if len(self.previous_boxes) > 0:
            im = np.stack((self.initial_frame, img))
            priori = np.stack((self.initial_boxes, self.previous_boxes))
            scores, new_boxes = self.network.track(im, priori)
            frame_size = (img.shape[1], img.shape[0])
            clip_box(new_boxes, frame_size)
            return scores, new_boxes
        else:
            return np.zeros(0), np.zeros((0, 4))


class ImageDirSet(data.Dataset):
    def __init__(self, image_dir, downsample=1):
        if os.path.isdir(image_dir):
            form = image_dir + '/*'
        else:
            form = image_dir + '*'
        self.images = sorted(glob.glob(form))[::downsample]
        assert len(self.images) >= 1, "Not enough images presented"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return cv2.imread(self.images[index])

    def __iter__(self):
        for i in range(len(self.images)):
            yield self.__getitem__(i)

    def get_wh(self):
        h, w, _ = self[0].shape
        return w, h


def multi_object_track(net, args):
    tracker_cls = MultiTracker
    interval = args.interval
    results = read_det_file(args.file)
    if args.write_results != '':
        res = open(args.write_results, 'w')
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    loader = ImageDirSet(args.image_dir)

    for i in range(args.start, len(loader)):
        print("Frame ", i)
        image = loader[i]
        im2show = image.copy()
        bboxes = np.array(results[0][i])
        scores = np.array(results[1][i])
        cls_inds = np.array(results[2][i])

        scores, bboxes, cls_inds = filter_byconf(scores, args.conf, bboxes, cls_inds)
        box_inds = list(map(str, range(len(bboxes))))

        if (i - args.start) % interval == 0:
            tracker = tracker_cls(net, image, bboxes)
            bboxes_pred = bboxes
            scores_pred = scores
            cls_inds_pred = cls_inds
            im2show = draw_detection(im2show, bboxes, scores_pred, range(
                len(bboxes)), box_inds, color=colors[1], width=1.0 / np.sqrt(1))
        else:
            scores_track, bboxes_pred = tracker.update(image)
            box_inds = list(map(str, range(len(bboxes_pred))))
            im2show = draw_detection(im2show, bboxes_pred, None, range(
                len(bboxes_pred)), box_inds, color=colors[0], width=1.0 / np.sqrt(2))

        if args.write_results != '':
            for bbox, score, cls in zip(bboxes_pred, scores_pred, cls_inds_pred):
                res.write('%d %d %.3f %.f %.f %.f %.f\n' %
                          ((i, cls, score) + tuple(bbox)))

        if args.visualize:
            cv2.imshow('win', im2show)
            cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('file')
    parser.add_argument('--interval', default=6, type=int,
                        help='Keyframe interval')
    parser.add_argument('--model', default='patchnet',
                        help='Tracker architecture')
    parser.add_argument('--pretrained', default='./pretrained/patchnet.pth')
    parser.add_argument('--conf', default=0.01, type=float,
                        help='Detection confidence threshold for tracking')

    parser.add_argument('--write-results', default='',
                        help='Store results to a file')
    parser.add_argument('--write-images', default='',
                        help='Store images to a director')
    parser.add_argument('--visualize', action='store_true',
                        help='Display tracking results')
    parser.add_argument('--start', type=int,
                        help='starting from Nth frame', default=0)
    args = parser.parse_args()

    if args.model == 'patchnet':
        net = PatchNet()
        net.create_architecture()
        if args.pretrained != '':
            net.load_state_dict(torch.load(args.pretrained))
    elif args.model == 'patchaggr':
        net = PatchAggregate()
        net.create_architecture()
        if args.pretrained != '':
            net.load_state_dict(torch.load(args.pretrained))
    else:
        raise NotImplementedError

    net.eval()
    net.cuda()

    multi_object_track(net, args)
