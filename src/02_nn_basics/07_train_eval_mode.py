#!/usr/bin/env python

"""train() vs eval() mode affects Dropout."""

import torch
from torch import nn

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
x = torch.ones(1, 4)

model.train()
print(model(x))
print(model(x))

model.eval()
print(model(x))
print(model(x))
