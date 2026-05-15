#!/usr/bin/env python

"""Stack multiple Transformer encoder layers."""

import torch
from torch import nn

torch.manual_seed(0)

layer = nn.TransformerEncoderLayer(
    d_model=16, nhead=4, dim_feedforward=64, batch_first=True,
)
encoder = nn.TransformerEncoder(layer, num_layers=3)

x = torch.randn(3, 8, 16)
print(encoder(x).shape)
