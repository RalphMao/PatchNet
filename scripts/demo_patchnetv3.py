from __future__ import absolute_import

import os
import sys
import torch
sys.path.append('.')
import glob
import numpy as np

from siamfc import TrackerSiamWithPatch, VariablePatchSiam
from patchnet import PatchNetv3


if __name__ == '__main__':
    seq_dir = os.path.expanduser('./data/OTB/Car1/')
    # seq_dir = os.path.expanduser('./data/OTB/Car4/')
    # seq_dir = os.path.expanduser('./data/OTB/Crowds/')
    # seq_dir = os.path.expanduser('./data/OTB/Ironman/')
    # seq_dir = os.path.expanduser('./data/OTB/Tiger1/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    try:
        anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    except:
        anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'ckpt/v3_all/patchnet_1.pth'
    patchnet = PatchNetv3()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path))
    patchnet.eval()
    patchnet.cuda()
    # tracker = TrackerSiamWithPatch(net_path, patchnet, 4)
    tracker = VariablePatchSiam(net_path, patchnet, interval=4)
    tracker.track(img_files, anno[0], visualize=True)
