# PyTorch-PatchNet

## Overview

This repo contains PatchNet model description in PyTorch and its training/evaluation with SiameseFC. It heavily relies on [siamfc-pytorch]([https://github.com/huanglianghua/siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)).

## Content

- [Demo](#Demo)
- [Prerequisite](#Prerequisite)
- [How to use](#How-to-use)


## Demo

<p align="center">
<img src=fig/surfer.gif>
</p>

All three methods are tested with PyTorch-1.4 on Nvidia Jetson Nano. 

## Prerequisite

* Python 3
* PyTorch >= 1.2
* GOT-10k dataset (for training)

## How to use

Both trained models of SiamFC and PatchNet have already been stored under ./pretrained/

**Demo** 
```
python3 scripts/demo_skipframe.py
```

**Evaluate on OTB2015 dataset**
```
python3 scripts/test_<name>.py
```

**Train PatchNet model**
```
python3 scripts/train_patchnet.py --dataset <got_10k_root> --save pretrained/<my_fancy_model>
```


