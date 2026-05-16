#!/usr/bin/env python

"""Adam optimizer on a 2-D quadratic."""

import torch


def main() -> torch.Tensor:
    p = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.1)

    for _ in range(200):
        opt.zero_grad()
        loss = (p[0] - 3) ** 2 + (p[1] + 4) ** 2
        loss.backward()
        opt.step()

    result = p.detach()
    print(result)
    return result


if __name__ == "__main__":
    main()
