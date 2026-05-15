#!/usr/bin/env python

"""One-layer affine normalizing flow."""

import math
import torch

torch.manual_seed(0)

s = torch.zeros(1, requires_grad=True)
t = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([s, t], lr=0.05)


def log_px(x: torch.Tensor) -> torch.Tensor:
    z = (x - t) / torch.exp(s)
    log_pz = -0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi)
    return log_pz - s


for _ in range(1000):
    samples = 2.0 * torch.randn(256, 1) + 3.0
    opt.zero_grad()
    loss = -log_px(samples).mean()
    loss.backward()
    opt.step()

print("s =", s.item(), "t =", t.item())
