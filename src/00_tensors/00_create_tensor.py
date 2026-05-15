#!/usr/bin/env python

"""Create a 1-D PyTorch tensor of integers 1..5."""

import torch

t = torch.tensor([1, 2, 3, 4, 5])
print(t)
print(t.dtype)
print(t.shape)
