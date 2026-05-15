#!/usr/bin/env python

"""Compare CE loss with and without label smoothing."""

import torch
from torch import nn

logits = torch.tensor([[5.0, 0.0, 0.0]])
target = torch.tensor([0])

print(nn.CrossEntropyLoss()(logits, target))
print(nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target))
