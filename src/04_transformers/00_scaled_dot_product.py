#!/usr/bin/env python

"""Scaled dot-product attention from scratch."""

import math
import torch


def main() -> torch.Tensor:
    torch.manual_seed(0)

    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v

    print(out.shape)
    return out


if __name__ == "__main__":
    main()
