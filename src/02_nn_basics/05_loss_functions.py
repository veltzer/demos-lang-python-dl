#!/usr/bin/env python

"""Three canonical losses."""

import torch
from torch import nn

mse = nn.MSELoss()
print(mse(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5])))

bce = nn.BCEWithLogitsLoss()
print(bce(torch.tensor([1.0, -1.0]), torch.tensor([1.0, 0.0])))

ce = nn.CrossEntropyLoss()
print(ce(torch.tensor([[2.0, 0.5, 1.0]]), torch.tensor([0])))
