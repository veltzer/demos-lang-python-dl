#!/usr/bin/env python

"""A single Transformer encoder layer."""

import torch
from torch import nn


def main() -> torch.Tensor:
    torch.manual_seed(0)

    layer = nn.TransformerEncoderLayer(
        d_model=16, nhead=4, dim_feedforward=64, batch_first=True,
    )
    x = torch.randn(3, 8, 16)
    out = layer(x)
    print(out.shape)
    return out


if __name__ == "__main__":
    main()
