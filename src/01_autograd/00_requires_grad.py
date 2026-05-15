#!/usr/bin/env python

"""Basic autograd on a scalar."""

import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
