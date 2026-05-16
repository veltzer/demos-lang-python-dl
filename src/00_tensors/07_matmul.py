#!/usr/bin/env python

"""Matrix multiplication three ways."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    c1 = torch.matmul(a, b)
    c2 = a @ b

    c3 = torch.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c3[i, j] += a[i, k] * b[k, j]

    print(c1)
    print(c2)
    print(c3)
    print(torch.allclose(c1, c3))
    return c1, c2, c3


if __name__ == "__main__":
    main()
