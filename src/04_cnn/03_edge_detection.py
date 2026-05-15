#!/usr/bin/env python

"""Detect a vertical edge with a hand-set Sobel-X conv."""

import torch
from torch import nn

conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]]]])
with torch.no_grad():
    conv.weight.copy_(sobel_x)

img = torch.zeros(1, 1, 5, 5)
img[:, :, :, 3:] = 1.0
print(img.squeeze())
print(conv(img).squeeze())
