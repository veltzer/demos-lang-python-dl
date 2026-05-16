#!/usr/bin/env python

"""Upsampling with ConvTranspose2d."""

import torch
from torch import nn


def main() -> torch.Tensor:
    up = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
    x = torch.randn(1, 8, 4, 4)
    y = up(x)
    print(y.shape)
    return y


if __name__ == "__main__":
    main()
