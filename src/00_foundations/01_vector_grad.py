#!/usr/bin/env python

"""Gradient of sum of squares."""

import torch


def main() -> torch.Tensor:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward()
    assert x.grad is not None
    print(x.grad)
    return x.grad


if __name__ == "__main__":
    main()
