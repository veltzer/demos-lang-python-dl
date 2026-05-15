#!/usr/bin/env python

"""Small CNN sized for MNIST-shape input."""

import torch
from torch import nn

torch.manual_seed(0)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10),
)

x = torch.randn(4, 1, 28, 28)
print(model(x).shape)
