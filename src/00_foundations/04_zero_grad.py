#!/usr/bin/env python

"""Gradient accumulation and zero_grad."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(2.0, requires_grad=True)

    y = x ** 2
    y.backward()
    assert x.grad is not None
    print(x.grad)
    g1 = x.grad.clone()

    y = x ** 2
    y.backward()
    print(x.grad)
    g2 = x.grad.clone()

    x.grad.zero_()
    y = x ** 2
    y.backward()
    print(x.grad)
    g3 = x.grad.clone()
    return g1, g2, g3


if __name__ == "__main__":
    main()
