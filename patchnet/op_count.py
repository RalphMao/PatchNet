import numpy as np
import torch

def count_fourierfeat(model, x, y):
    x, z = x
    b, k, kx, ky = z.shape
    B, K, H, W = x.shape
    _, Cout, Hout, Wout = y.shape
    assert b == B and k == K

    fft_ops = 2 * b * k * kx * ky * np.log2(kx) * np.log2(ky)
    corr_ops = B * Cout * Hout * Wout * model.k * model.k * K
    model.total_ops = torch.Tensor([corr_ops + fft_ops])
    # model.total_ops = torch.Tensor([0])
    
def count_poolsoftselect(model, x, y):
    score, offset = x
    sp, op = y
    n, c, H, W = score.shape
    ks, stride = model.ks, model.stride
    ratio = float(ks * ks) / stride / stride
    ops = (score.nelement() + offset.nelement()) * (2 + ratio) 
    model.total_ops = torch.Tensor([int(ops)])
    # model.total_ops = torch.Tensor([0])
