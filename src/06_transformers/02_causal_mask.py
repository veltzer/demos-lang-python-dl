#!/usr/bin/env python

"""Causal masking in scaled dot-product attention."""

import math
import torch

torch.manual_seed(0)

seq = 4
d = 8
q = torch.randn(1, seq, d)
k = torch.randn(1, seq, d)
v = torch.randn(1, seq, d)

scores = q @ k.transpose(-2, -1) / math.sqrt(d)
mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)
print(weights.squeeze())
