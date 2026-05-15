#!/usr/bin/env python

"""BatchNorm2d normalizes per channel."""

import torch
from torch import nn

torch.manual_seed(0)

bn = nn.BatchNorm2d(4)
x = torch.randn(8, 4, 16, 16) * 5 + 10
y = bn(x)

print("mean per channel:", y.mean(dim=(0, 2, 3)))
print("std  per channel:", y.std(dim=(0, 2, 3), unbiased=False))
