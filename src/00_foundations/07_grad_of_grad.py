#!/usr/bin/env python

"""Second-order gradient via create_graph."""

import torch


def main() -> tuple[float, float]:
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3

    (grad1,) = torch.autograd.grad(y, x, create_graph=True)
    (grad2,) = torch.autograd.grad(grad1, x)

    print(grad1.item(), grad2.item())
    return grad1.item(), grad2.item()


if __name__ == "__main__":
    main()
