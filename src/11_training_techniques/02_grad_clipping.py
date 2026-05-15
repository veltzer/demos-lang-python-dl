#!/usr/bin/env python

"""Gradient clipping bounds the total gradient norm."""

import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
for p in model.parameters():
    p.grad = torch.ones_like(p) * 1e6

before = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
after = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5

print("before:", before.item())
print("after :", after.item())
