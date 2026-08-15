#!/usr/bin/env python

"""Gradient clipping bounds the total gradient norm."""

import torch
from torch import nn


def main() -> tuple[float, float]:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 1e6

    def total_norm() -> torch.Tensor:
        sq = torch.zeros(())
        for param in model.parameters():
            assert param.grad is not None
            sq = sq + param.grad.norm() ** 2
        return sq ** 0.5

    before_v = total_norm().item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    after_v = total_norm().item()
    print("before:", before_v)
    print("after :", after_v)
    return before_v, after_v


if __name__ == "__main__":
    main()
