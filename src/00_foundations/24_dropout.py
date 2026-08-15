#!/usr/bin/env python

"""Dropout is stochastic in train mode."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )

    x = torch.randn(1, 10)

    model.train()
    t1 = model(x)
    t2 = model(x)
    print("train run 1:", t1)
    print("train run 2:", t2)

    model.eval()
    e1 = model(x)
    e2 = model(x)
    print("eval  run 1:", e1)
    print("eval  run 2:", e2)
    return t1, t2, e1, e2


if __name__ == "__main__":
    main()
