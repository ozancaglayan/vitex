#!/usr/bin/env python

import time
import sys
import numpy as np
import torch

from src.utils import np2ascii


if __name__ == '__main__':
    data = np.load('train-fs12-mh12.npz')
    slices = data['lens']
    data = data['data']

    x = data[slices[0][0]: slices[0][1]]
    x = torch.from_numpy(x)
    patches = x.view((12, -1)).unfold(1, 50, 15).permute(1, 0, 2)

    for patch in patches:
        print("\x1b[2J\x1b[H")
        np2ascii(patch)
        time.sleep(0.5)
