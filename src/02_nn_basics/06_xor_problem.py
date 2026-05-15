#!/usr/bin/env python

"""Train a tiny MLP to compute XOR."""

import torch
from torch import nn

torch.manual_seed(0)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.BCEWithLogitsLoss()

for _ in range(5000):
    logits = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    preds = torch.sigmoid(model(x)).round()
print(preds.squeeze().tolist())
