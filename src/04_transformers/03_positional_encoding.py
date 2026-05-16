#!/usr/bin/env python

"""Sinusoidal positional encoding table."""

import math
import torch


def positional_encoding(max_len: int, d: int) -> torch.Tensor:
    table = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


def main() -> torch.Tensor:
    pe = positional_encoding(20, 16)
    print(pe.shape)
    print(pe[0])
    return pe


if __name__ == "__main__":
    main()
