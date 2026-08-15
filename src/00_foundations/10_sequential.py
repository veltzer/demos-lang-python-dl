#!/usr/bin/env python

"""Compose layers with nn.Sequential."""

import torch
from torch import nn


def main() -> tuple[nn.Sequential, torch.Tensor]:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )

    x = torch.randn(8, 10)
    y = model(x)
    print(y.shape)
    return model, y


if __name__ == "__main__":
    main()
