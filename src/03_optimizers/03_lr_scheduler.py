#!/usr/bin/env python

"""StepLR halves the learning rate periodically."""

import torch
from torch import nn

model = nn.Linear(2, 2)
opt = torch.optim.Adam(model.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

for epoch in range(20):
    opt.step()
    sched.step()
    print(epoch, opt.param_groups[0]["lr"])
