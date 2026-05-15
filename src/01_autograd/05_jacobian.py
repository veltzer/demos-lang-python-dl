#!/usr/bin/env python

"""Computing a Jacobian with torch.autograd.functional."""

import torch
from torch.autograd.functional import jacobian


def f(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([v[0] ** 2, v[0] * v[1], v[1] ** 3])


x = torch.tensor([2.0, 3.0])
print(jacobian(f, x))
