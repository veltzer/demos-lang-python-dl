#!/usr/bin/env python

"""Common activation functions side by side."""

import torch
from torch import nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print("relu:   ", nn.ReLU()(x))
print("sigmoid:", nn.Sigmoid()(x))
print("tanh:   ", nn.Tanh()(x))
print("softmax:", nn.Softmax(dim=-1)(x))
