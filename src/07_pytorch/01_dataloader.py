#!/usr/bin/env python

"""DataLoader produces mini-batches."""

import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
ds = TensorDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
loader = DataLoader(ds, batch_size=16, shuffle=True)

for i, (xb, yb) in enumerate(loader):
    print(i, xb.shape, yb.shape)
    if i == 2:
        break
