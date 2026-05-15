#!/usr/bin/env python

"""Forecast next value of a sine wave with an LSTM."""

import math
import torch
from torch import nn

torch.manual_seed(0)

t = torch.linspace(0, 8 * math.pi, 200)
y = torch.sin(t)

window = 20
xs, ys = [], []
for i in range(len(y) - window):
    xs.append(y[i:i + window])
    ys.append(y[i + window])
x = torch.stack(xs).unsqueeze(-1)
target = torch.stack(ys).unsqueeze(-1)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(inp)
        return self.head(out[:, -1])


model = Model()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(300):
    opt.zero_grad()
    loss = loss_fn(model(x), target)
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(epoch, loss.item())
