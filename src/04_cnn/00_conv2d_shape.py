#!/usr/bin/env python

"""Shape of a Conv2d output with same-padding."""

import torch
from torch import nn

torch.manual_seed(0)

conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
x = torch.randn(2, 3, 32, 32)
print(conv(x).shape)
