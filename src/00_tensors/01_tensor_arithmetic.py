#!/usr/bin/env python

"""Elementwise and dot operations on tensors."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([10, 20, 30])

    s = a + b
    p = a * b
    d = torch.dot(a, b)

    print(s)
    print(p)
    print(d)
    return s, p, d


if __name__ == "__main__":
    main()
