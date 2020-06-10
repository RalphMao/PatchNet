# PyTorch-PatchNet

This repo contains PatchNet model description in PyTorch and its training/evaluation with SiameseFC. It heavily relies on [siamfc-pytorch]([https://github.com/huanglianghua/siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)).

## Prerequisite

* Python 3
* PyTorch>1.2
* GOT-10k dataset (for training)

## How to use

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
