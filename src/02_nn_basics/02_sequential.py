#!/usr/bin/env python

"""Compose layers with nn.Sequential."""

import torch
from torch import nn

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

x = torch.randn(8, 10)
print(model(x).shape)
