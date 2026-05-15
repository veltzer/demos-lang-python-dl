#!/usr/bin/env python

"""Minimize (x - 7)^2 with SGD."""

import torch

x = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([x], lr=0.1)

for _ in range(100):
    opt.zero_grad()
    loss = (x - 7) ** 2
    loss.backward()
    opt.step()

print(x.item())
