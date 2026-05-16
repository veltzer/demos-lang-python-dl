#!/usr/bin/env python

"""LSTM output and (h, c) shapes."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    lstm = nn.LSTM(4, 6, num_layers=2, batch_first=True)
    x = torch.randn(2, 7, 4)
    out, (h, c) = lstm(x)
    print(out.shape)
    print(h.shape)
    print(c.shape)
    return out, h, c


if __name__ == "__main__":
    main()
