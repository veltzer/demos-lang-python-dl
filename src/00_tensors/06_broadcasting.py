#!/usr/bin/env python

"""Broadcasting a row vector across a matrix."""

import torch

m = torch.arange(12).view(3, 4)
v = torch.arange(4) * 10

print(m)
print(v)
print(m + v)
