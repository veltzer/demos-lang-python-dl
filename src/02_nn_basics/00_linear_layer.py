#!/usr/bin/env python

"""A single nn.Linear layer."""

import torch
from torch import nn

torch.manual_seed(0)

layer = nn.Linear(3, 2)
x = torch.randn(4, 3)
y = layer(x)

print(y.shape)
print(layer.weight.shape)
print(layer.bias.shape)
