#!/usr/bin/env python

"""Generate random tensors with a fixed seed."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    u = torch.rand(2, 3)
    n = torch.randn(2, 3)
    i = torch.randint(0, 10, (5,))

    print(u)
    print(n)
    print(i)
    return u, n, i


if __name__ == "__main__":
    main()
