#!/usr/bin/env python

"""Tiny denoising diffusion model on 1-D N(5, 1) data."""

import torch
from torch import nn


T = 20
alphas = torch.linspace(0.99, 0.05, T)


class EpsNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_t: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t_idx.float() / T], dim=1))


def main() -> float:
    torch.manual_seed(0)
    model = EpsNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = torch.tensor(0.0)
    for _ in range(2000):
        x0 = torch.randn(128, 1) + 5
        t = torch.randint(0, T, (128, 1))
        a = alphas[t.squeeze()].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps

        opt.zero_grad()
        loss = ((model(xt, t) - eps) ** 2).mean()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    print(final_loss)
    return final_loss


if __name__ == "__main__":
    main()
