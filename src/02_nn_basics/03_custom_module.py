#!/usr/bin/env python

"""Custom nn.Module subclass."""

import torch
from torch import nn


class TwoLayerNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


torch.manual_seed(0)
model = TwoLayerNet(20, 64, 3)
print(model(torch.randn(5, 20)).shape)
