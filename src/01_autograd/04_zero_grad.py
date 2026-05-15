#!/usr/bin/env python

"""Gradient accumulation and zero_grad."""

import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2
y.backward()
print(x.grad)

y = x ** 2
y.backward()
print(x.grad)

x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)
