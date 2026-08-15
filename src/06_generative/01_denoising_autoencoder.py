#!/usr/bin/env python

"""Train an autoencoder to denoise its input."""

import torch
from torch import nn


def main() -> float:
    torch.manual_seed(0)

    x = torch.randn(200, 8)

    model = nn.Sequential(nn.Linear(8, 2), nn.Linear(2, 8))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(1000):
        noisy = x + 0.3 * torch.randn_like(x)
        opt.zero_grad()
        loss = loss_fn(model(noisy), x)
        loss.backward()
        opt.step()

    with torch.no_grad():
        clean_loss = loss_fn(model(x), x).item()
    print("denoising loss on clean input:", clean_loss)
    return clean_loss


if __name__ == "__main__":
    main()
