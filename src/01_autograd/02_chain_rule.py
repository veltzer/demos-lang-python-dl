#!/usr/bin/env python

"""Chain rule through a small computation graph."""

import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y * y + y
z.backward()
print(x.grad)
