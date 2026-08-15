#!/usr/bin/env python

"""Stack multiple Transformer encoder layers."""

import torch
from torch import nn


def main() -> torch.Tensor:
    torch.manual_seed(0)

    layer = nn.TransformerEncoderLayer(
        d_model=16, nhead=4, dim_feedforward=64, batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=3)

    x = torch.randn(3, 8, 16)
    out = encoder(x)
    print(out.shape)
    return out


if __name__ == "__main__":
    main()
