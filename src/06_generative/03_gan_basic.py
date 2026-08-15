#!/usr/bin/env python

"""Tiny 1-D GAN learning N(5, 1)."""

import torch
from torch import nn


def real_samples(n: int) -> torch.Tensor:
    return torch.randn(n, 1) + 5


def noise(n: int) -> torch.Tensor:
    return torch.randn(n, 1)


def main() -> tuple[float, float]:
    torch.manual_seed(0)

    gen = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    disc = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(2000):
        real = real_samples(128)
        fake = gen(noise(128)).detach()
        opt_d.zero_grad()
        loss_d = bce(disc(real), torch.ones(128, 1)) + bce(disc(fake), torch.zeros(128, 1))
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad()
        fake = gen(noise(128))
        loss_g = bce(disc(fake), torch.ones(128, 1))
        loss_g.backward()
        opt_g.step()

    with torch.no_grad():
        samples = gen(noise(1000))
    mean = samples.mean().item()
    std = samples.std().item()
    print("mean:", mean, "std:", std)
    return mean, std


if __name__ == "__main__":
    main()
