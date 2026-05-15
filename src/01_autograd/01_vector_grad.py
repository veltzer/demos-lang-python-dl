#!/usr/bin/env python

"""Gradient of sum of squares."""

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(x.grad)
