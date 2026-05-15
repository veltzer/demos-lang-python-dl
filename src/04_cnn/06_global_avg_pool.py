#!/usr/bin/env python

"""Global average pooling via AdaptiveAvgPool2d."""

import torch
from torch import nn

x = torch.randn(2, 16, 8, 8)
pool = nn.AdaptiveAvgPool2d(1)
y = pool(x).squeeze(-1).squeeze(-1)
print(y.shape)
