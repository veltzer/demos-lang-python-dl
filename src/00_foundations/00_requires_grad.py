#!/usr/bin/env python

"""Basic autograd on a scalar."""

import torch


def main() -> torch.Tensor:
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    assert x.grad is not None
    print(x.grad)
    return x.grad


if __name__ == "__main__":
    main()
