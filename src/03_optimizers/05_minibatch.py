#!/usr/bin/env python

"""Mini-batch SGD with manual shuffling."""

import torch
from torch import nn


def main() -> tuple[float, float]:
    torch.manual_seed(0)

    n = 200
    x = torch.rand(n, 1) * 10
    y = 2 * x + 1 + 0.5 * torch.randn(n, 1)

    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    batch = 16
    for _ in range(20):
        idx = torch.randperm(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            opt.zero_grad()
            loss_fn(model(x[b]), y[b]).backward()
            opt.step()

    w_val = model.weight.item()
    b_val = model.bias.item()
    print("w =", w_val)
    print("b =", b_val)
    return w_val, b_val


if __name__ == "__main__":
    main()
