#!/usr/bin/env python

"""Linear regression via gradient descent."""

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

    for _ in range(1000):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    w = model.weight.item()
    b = model.bias.item()
    print("w =", w)
    print("b =", b)
    return w, b


if __name__ == "__main__":
    main()
