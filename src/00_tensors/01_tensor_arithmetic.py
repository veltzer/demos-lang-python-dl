#!/usr/bin/env python

"""Elementwise and dot operations on tensors."""

import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])

print(a + b)
print(a * b)
print(torch.dot(a, b))
