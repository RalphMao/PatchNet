from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset


__all__ = ['Pair']


class Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1, max_distance=100,
                 fixed=False):
        super(Pair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.max_distance = max_distance
        self.indices = np.random.permutation(len(seqs))
        self.return_meta = getattr(seqs, 'return_meta', False)
        self.fixed = fixed

    def __getitem__(self, index_original):
        index = self.indices[index_original % len(self.indices)]
        fold = index_original // len(self.indices)

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        if self.fixed:
            dist = fold % self.max_distance + 1
            zidx = fold * ((len(val_indices) - dist) // self.pairs_per_seq)
            zidx = max(0, zidx)
            xidx = zidx + dist
            xidx = min(xidx, len(val_indices) - 1)
            rand_z = val_indices[zidx]
            rand_x = val_indices[xidx]

        else:
            rand_z, rand_x = self._sample_pair(val_indices)

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        if x.shape != z.shape:
            print("Warning: shape mismatch for %s and %s"%(
                    img_files[rand_z], img_files[rand_x]))
            z = x
            box_z = box_x

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)
        
        return item
    
    def __len__(self):
        return len(self.indices) * self.pairs_per_seq
    
    def _sample_pair(self, indices):
        n = len(indices)
        num_inds = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                zindex = np.random.randint(num_inds)
                xindex = (zindex + np.random.randint(self.max_distance)) % num_inds
                rand_z = indices[zindex]
                rand_x = indices[xindex]
                if np.abs(rand_x - rand_z) < self.max_distance:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z
            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
