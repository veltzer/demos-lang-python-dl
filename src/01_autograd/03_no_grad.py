#!/usr/bin/env python

"""Disabling gradient tracking."""

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

with torch.no_grad():
    y = x * 2
print("no_grad:", y.requires_grad)

z = x.detach() * 2
print("detach:", z.requires_grad)
