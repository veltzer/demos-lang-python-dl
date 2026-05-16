#!/usr/bin/env python

"""Chain rule through a small computation graph."""

import torch


def main() -> torch.Tensor:
    x = torch.tensor(2.0, requires_grad=True)
    y = x * 3
    z = y * y + y
    z.backward()
    assert x.grad is not None
    print(x.grad)
    return x.grad


if __name__ == "__main__":
    main()
