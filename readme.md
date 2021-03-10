# PyTorch-PatchNet

## Overview

This repo contains PatchNet model description in PyTorch and its training/evaluation with SiameseFC. It heavily relies on [siamfc-pytorch]([https://github.com/huanglianghua/siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)).

- [Demo](#Demo)
- [Prerequisite](#Prerequisite)
- [How to use](#How-to-use)


## Demo

<p align="center">
<img src=fig/surfer.gif>
</p>

From left to right:
Skip frame - interleave siamfc with patchnet for speedup.
Boosted - combine siamfc's prediction scores and patchnet's local similarity scores for more robust tracking.
SiamFC - the base line.

All three methods are tested with PyTorch-1.4 on Nvidia Jetson Nano. 

## Prerequisite

* Python 3
* PyTorch >= 1.2
* GOT-10k dataset (for training only)

## How to use

Both trained models of SiamFC and PatchNet have already been stored under ./pretrained/

**Demo** 
```
python3 scripts/demo_skipframe.py
```

**Interleave SiamFC and PatchNet on OTB2015 dataset**
```
python3 scripts/test_<name>.py
```

**Interleave RFCN ResNet-101 and PatchNet**
```
python3 scripts/vod_track.py data/VID/ILSVRC2015_val_00142000 data/VID/rfcn_res101_val_00142000.txt --visualize --conf 0.5
```

**Train PatchNet model**
```
python3 scripts/train_patchnet.py --dataset <got_10k_root> --save pretrained/<my_fancy_model>
```


