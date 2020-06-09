from __future__ import absolute_import

import os
import sys
import torch
sys.path.append('.')
from got10k.experiments import *

from siamfc import TrackerCombined
from patchnet import PatchNet


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'pretrained/patchnet.pth'
    patchnet = PatchNet()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path))
    patchnet.eval()
    patchnet.cuda()
    # tracker = TrackerSiamWithPatch(net_path, patchnet, 4, 0.35)
    tracker = TrackerCombined(net_path, patchnet, factor=0.05)

    root_dir = os.path.expanduser('./data/OTB')
    e = ExperimentOTB(root_dir, version=2015)
    tracker.name = "Combined-0.05"
    e.run(tracker)
    e.report([tracker.name])
