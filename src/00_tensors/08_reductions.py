#!/usr/bin/env python

"""Reductions over tensor axes."""

import torch

torch.manual_seed(0)
t = torch.rand(3, 4)

print(t.sum())
print(t.sum(dim=0))
print(t.mean(dim=1))
print(t.argmax())
