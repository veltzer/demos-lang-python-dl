#!/usr/bin/env python

"""A linear autoencoder learning to reconstruct its input."""

import torch
from torch import nn

torch.manual_seed(0)

x = torch.randn(200, 8)

model = nn.Sequential(nn.Linear(8, 2), nn.Linear(2, 8))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(1000):
    opt.zero_grad()
    loss = loss_fn(model(x), x)
    loss.backward()
    opt.step()

print(loss.item())
