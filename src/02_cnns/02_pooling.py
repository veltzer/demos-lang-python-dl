#!/usr/bin/env python

"""MaxPool vs AvgPool."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    maxp = nn.MaxPool2d(2, 2)(x)
    avgp = nn.AvgPool2d(2, 2)(x)
    print(x)
    print(maxp)
    print(avgp)
    return x, maxp, avgp


if __name__ == "__main__":
    main()
