#!/usr/bin/env python

"""A minimal variational autoencoder."""

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_dim: int = 8, latent: int = 4) -> None:
        super().__init__()
        self.enc = nn.Linear(in_dim, latent * 2)
        self.dec = nn.Linear(latent, in_dim)
        self.latent = latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.dec(z), mu, logvar


def main() -> float:
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = torch.tensor(0.0)
    for _ in range(1000):
        opt.zero_grad()
        recon, mu, logvar = model(x)
        recon_loss = ((recon - x) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        loss = recon_loss + kl
        loss.backward()
        opt.step()

    final_loss = loss.item()
    print(final_loss)
    return final_loss


if __name__ == "__main__":
    main()
