from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
sys.path.append('.')
from got10k.experiments import *

from siamfc import TrackerSiamWithPatch, VariablePatchSiam, VariableMultiPatchSiam
from patchnet import PatchNetv3


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'ckpt/v3_all/patchnet_1.pth'
    patchnet = PatchNetv3()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path))
    patchnet.eval()
    patchnet.cuda()
    # tracker = TrackerSiamWithPatch(net_path, patchnet, 4, 0.35)
    # tracker = VariablePatchSiam(net_path, patchnet, interval=4)
    tracker = VariableMultiPatchSiam(net_path, patchnet, interval=4)

    root_dir = os.path.expanduser('./data/OTB')
    e = ExperimentOTB(root_dir, version=2015)
    tracker.name = "PatchNetv3-multi-variable-4-0.8-2"
    e.run(tracker)
    e.report([tracker.name])
    print("Overall track len: %f"%(np.mean(tracker.tracklen_counter)))
