#!/usr/bin/env python

"""MaxPool vs AvgPool."""

import torch
from torch import nn

x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
print(x)
print(nn.MaxPool2d(2, 2)(x))
print(nn.AvgPool2d(2, 2)(x))
