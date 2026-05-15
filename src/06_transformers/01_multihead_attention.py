#!/usr/bin/env python

"""Multi-head self-attention via nn.MultiheadAttention."""

import torch
from torch import nn

torch.manual_seed(0)

mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
x = torch.randn(2, 6, 16)
out, weights = mha(x, x, x)
print(out.shape)
print(weights.shape)
