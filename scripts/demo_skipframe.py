from __future__ import absolute_import

import os
import sys
import torch
sys.path.append('.')
import glob
import numpy as np

from siamfc import VariablePatchSiam
from patchnet import PatchNet


if __name__ == '__main__':
    seq_dir = os.path.expanduser('./data/OTB/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    try:
        anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    except:
        anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'pretrained/patchnet.pth'
    patchnet = PatchNet()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path))
    patchnet.eval()
    patchnet.cuda()
    tracker = VariablePatchSiam(net_path, patchnet, interval=4)
    tracker.track(img_files, anno[0], visualize=True)
