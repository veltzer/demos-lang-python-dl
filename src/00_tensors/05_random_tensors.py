#!/usr/bin/env python

"""Generate random tensors with a fixed seed."""

import torch

torch.manual_seed(42)

print(torch.rand(2, 3))
print(torch.randn(2, 3))
print(torch.randint(0, 10, (5,)))
