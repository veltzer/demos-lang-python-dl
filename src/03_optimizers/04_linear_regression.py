#!/usr/bin/env python

"""Linear regression via gradient descent."""

import torch
from torch import nn

torch.manual_seed(0)

n = 200
x = torch.rand(n, 1) * 10
y = 2 * x + 1 + 0.5 * torch.randn(n, 1)

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(1000):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()

print("w =", model.weight.item())
print("b =", model.bias.item())
