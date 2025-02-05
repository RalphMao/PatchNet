from __future__ import absolute_import

import os
import sys
import argparse
import torch
from got10k.datasets import *
sys.path.append('.')

from siamfc import TrainerPatchNet


if __name__ == '__main__':
    # root_dir = os.path.expanduser('~/data/GOT-10k')
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--dataset', default='/ssd2/dataset/got-10')
    parser.add_argument('--bs', dest='batch_size', type=int, default=1)
    parser.add_argument('--epoch', dest='epoch_num', type=int, default=10)
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--save', default='pretrained')
    args = parser.parse_args()

    root_dir = args.dataset
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    val_seqs = GOT10k(root_dir, subset='val', return_meta=True)

    tracker = TrainerPatchNet()
    if args.pretrained != '':
        tracker.net.load_state_dict(torch.load(args.pretrained))
    tracker.train_over(seqs, args, val_seqs, save_dir=args.save)
