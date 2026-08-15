#!/usr/bin/env python

"""Bidirectional GRU shapes."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    gru = nn.GRU(3, 5, bidirectional=True, batch_first=True)
    x = torch.randn(1, 6, 3)
    out, h = gru(x)
    print(out.shape)
    print(h.shape)
    return out, h


if __name__ == "__main__":
    main()
