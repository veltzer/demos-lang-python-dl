#!/usr/bin/env python

"""Sanity check: model can drive loss to ~0 on a tiny fixed batch."""

import torch
from torch import nn

torch.manual_seed(0)

x = torch.randn(8, 4)
y = torch.randint(0, 3, (8,))

model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for step in range(500):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(step, loss.item())
