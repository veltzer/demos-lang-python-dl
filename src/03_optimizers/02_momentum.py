#!/usr/bin/env python

"""SGD with and without momentum on a non-convex function."""

import torch


def f(x: torch.Tensor) -> torch.Tensor:
    return x ** 4 - 3 * x ** 2 + 2


def run(use_momentum: bool) -> float:
    x = torch.tensor(2.0, requires_grad=True)
    momentum = 0.9 if use_momentum else 0.0
    opt = torch.optim.SGD([x], lr=0.01, momentum=momentum)
    for _ in range(30):
        opt.zero_grad()
        f(x).backward()
        opt.step()
    return x.item()


print("plain   :", run(False))
print("momentum:", run(True))
