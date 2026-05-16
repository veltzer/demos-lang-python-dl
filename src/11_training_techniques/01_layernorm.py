#!/usr/bin/env python

"""LayerNorm normalizes within each sample."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    ln = nn.LayerNorm(4)
    x = torch.randn(2, 4) * 5 + 3
    y = ln(x)
    means = y.mean(dim=1)
    stds = y.std(dim=1, unbiased=False)
    print("mean per row:", means)
    print("std  per row:", stds)
    return means, stds


if __name__ == "__main__":
    main()
