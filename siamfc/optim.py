import torch

class AdamSparse(torch.optim.Adam):
    def __init__(self, params, *args, **kwargs):
        super(AdamSparse, self).__init__(params, *args, **kwargs)
        self.mask_groups = self._get_mask(self.param_groups)

    def _get_mask(self, param_groups):
        mask_groups = []
        for group in param_groups:
            mask_group = []
            for p in group['params']:
                mask_group.append((torch.abs(p) > 1e-9).float())
            mask_groups.append(mask_group)
        return mask_groups

    def step(self, closure=None):
        for group, mask_group in zip(self.param_groups, self.mask_groups):
            for p, mask in zip(group['params'], mask_group):
                if p.grad is None:
                    continue
                p.grad *= mask
        super(AdamSparse, self).step(closure=closure)
