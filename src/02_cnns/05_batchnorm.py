#!/usr/bin/env python

"""BatchNorm2d normalizes per channel."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    bn = nn.BatchNorm2d(4)
    x = torch.randn(8, 4, 16, 16) * 5 + 10
    y = bn(x)

    mean = y.mean(dim=(0, 2, 3))
    std = y.std(dim=(0, 2, 3), unbiased=False)
    print("mean per channel:", mean)
    print("std  per channel:", std)
    return mean, std


if __name__ == "__main__":
    main()
