#!/usr/bin/env python

"""Effect of stride and padding on Conv2d output size."""

import torch
from torch import nn

x = torch.randn(1, 1, 7, 7)
for stride in (1, 2):
    for padding in (0, 1):
        conv = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding)
        print(f"stride={stride} pad={padding} -> {conv(x).shape}")
