#!/usr/bin/env python

"""Plain gradient descent without torch.optim."""

import torch

x = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for _ in range(50):
    loss = (x - 5) ** 2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

print(x.item())
