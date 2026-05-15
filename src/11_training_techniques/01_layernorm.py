#!/usr/bin/env python

"""LayerNorm normalizes within each sample."""

import torch
from torch import nn

torch.manual_seed(0)

ln = nn.LayerNorm(4)
x = torch.randn(2, 4) * 5 + 3
y = ln(x)
print("mean per row:", y.mean(dim=1))
print("std  per row:", y.std(dim=1, unbiased=False))
