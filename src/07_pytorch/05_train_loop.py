#!/usr/bin/env python

"""Canonical PyTorch training loop."""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

n = 500
x = torch.randn(n, 4)
w = torch.randn(4, 3)
y = (x @ w).argmax(dim=1)

loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0.0
    correct = 0
    seen = 0
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += len(xb)
    print(f"epoch {epoch}: loss={total_loss / seen:.4f} acc={correct / seen:.4f}")
