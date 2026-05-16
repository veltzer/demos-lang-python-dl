#!/usr/bin/env python

"""Effect of stride and padding on Conv2d output size."""

import torch
from torch import nn


def main() -> list[tuple[int, int, torch.Size]]:
    x = torch.randn(1, 1, 7, 7)
    results: list[tuple[int, int, torch.Size]] = []
    for stride in (1, 2):
        for padding in (0, 1):
            conv = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding)
            shape = conv(x).shape
            results.append((stride, padding, shape))
            print(f"stride={stride} pad={padding} -> {shape}")
    return results


if __name__ == "__main__":
    main()
