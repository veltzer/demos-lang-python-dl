#!/usr/bin/env python

"""How weight_decay shrinks parameters."""

import torch
from torch import nn


def fit(weight_decay: float, x: torch.Tensor, y: torch.Tensor) -> float:
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(1000):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    return model.weight.item()


def main() -> tuple[float, float]:
    torch.manual_seed(0)
    x = torch.rand(200, 1) * 10
    y = 2 * x + 1 + 0.5 * torch.randn(200, 1)

    no_decay = fit(0.0, x, y)
    with_decay = fit(0.5, x, y)
    print("no decay :", no_decay)
    print("with decay:", with_decay)
    return no_decay, with_decay


if __name__ == "__main__":
    main()
