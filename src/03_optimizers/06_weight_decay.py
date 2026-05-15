#!/usr/bin/env python

"""How weight_decay shrinks parameters."""

import torch
from torch import nn

torch.manual_seed(0)
x = torch.rand(200, 1) * 10
y = 2 * x + 1 + 0.5 * torch.randn(200, 1)


def fit(weight_decay: float) -> float:
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(1000):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    return model.weight.item()


print("no decay :", fit(0.0))
print("with decay:", fit(0.5))
