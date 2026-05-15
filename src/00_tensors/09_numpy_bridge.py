#!/usr/bin/env python

"""NumPy <-> PyTorch zero-copy bridge."""

import numpy as np
import torch

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

t.mul_(2)

print("numpy:", arr)
print("tensor:", t)
print("back to numpy:", t.numpy())
