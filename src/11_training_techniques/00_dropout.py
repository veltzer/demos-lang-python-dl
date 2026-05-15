#!/usr/bin/env python

"""Dropout is stochastic in train mode."""

import torch
from torch import nn

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2),
)

x = torch.randn(1, 10)

model.train()
print("train run 1:", model(x))
print("train run 2:", model(x))

model.eval()
print("eval  run 1:", model(x))
print("eval  run 2:", model(x))
