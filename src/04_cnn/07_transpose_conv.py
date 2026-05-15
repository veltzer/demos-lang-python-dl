#!/usr/bin/env python

"""Upsampling with ConvTranspose2d."""

import torch
from torch import nn

up = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
x = torch.randn(1, 8, 4, 4)
print(up(x).shape)
