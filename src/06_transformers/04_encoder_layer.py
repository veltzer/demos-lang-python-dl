#!/usr/bin/env python

"""A single Transformer encoder layer."""

import torch
from torch import nn

torch.manual_seed(0)

layer = nn.TransformerEncoderLayer(
    d_model=16, nhead=4, dim_feedforward=64, batch_first=True,
)
x = torch.randn(3, 8, 16)
print(layer(x).shape)
