#!/usr/bin/env python

"""Indexing and slicing a 5x5 tensor."""

import torch

t = torch.arange(25).view(5, 5)

print(t[1])
print(t[:, -1])
print(t[1:4, 1:4])
