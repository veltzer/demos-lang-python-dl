#!/usr/bin/env python

"""StepLR halves the learning rate periodically."""

import torch
from torch import nn


def main() -> list[float]:
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    lrs = []
    for epoch in range(20):
        opt.step()
        sched.step()
        lr = opt.param_groups[0]["lr"]
        lrs.append(lr)
        print(epoch, lr)
    return lrs


if __name__ == "__main__":
    main()
