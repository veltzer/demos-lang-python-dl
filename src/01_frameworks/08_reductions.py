#!/usr/bin/env python

"""Reductions over tensor axes."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    t = torch.rand(3, 4)

    s = t.sum()
    cs = t.sum(dim=0)
    rm = t.mean(dim=1)
    am = t.argmax()

    print(s)
    print(cs)
    print(rm)
    print(am)
    return s, cs, rm, am


if __name__ == "__main__":
    main()
