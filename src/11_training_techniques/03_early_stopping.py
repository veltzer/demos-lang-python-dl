#!/usr/bin/env python

"""Stop training when validation loss plateaus."""

import torch
from torch import nn

torch.manual_seed(0)

n = 400
x = torch.randn(n, 4)
w_true = torch.randn(4, 1)
y = x @ w_true + 0.1 * torch.randn(n, 1)

x_train, x_val = x[:300], x[300:]
y_train, y_val = y[:300], y[300:]

model = nn.Linear(4, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

best, patience, wait, stopped_at = float("inf"), 3, 0, 0
for epoch in range(200):
    opt.zero_grad()
    loss_fn(model(x_train), y_train).backward()
    opt.step()

    with torch.no_grad():
        val = loss_fn(model(x_val), y_val).item()

    if val < best - 1e-4:
        best = val
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            stopped_at = epoch
            break

print("stopped at epoch:", stopped_at)
print("best val loss   :", best)
