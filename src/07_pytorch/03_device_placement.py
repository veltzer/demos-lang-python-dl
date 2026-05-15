#!/usr/bin/env python

"""Move model and data to the available device."""

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = nn.Linear(4, 2).to(device)
x = torch.randn(3, 4).to(device)
y = model(x)
print(y.device)
