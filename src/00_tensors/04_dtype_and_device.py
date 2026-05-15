#!/usr/bin/env python

"""Tensor dtype conversion and device inspection."""

import torch

t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(t, t.dtype)
print(t.double(), t.double().dtype)
print(t.to(torch.int32), t.to(torch.int32).dtype)
print("device:", t.device)
