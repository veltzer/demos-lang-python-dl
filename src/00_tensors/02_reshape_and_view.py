#!/usr/bin/env python

"""Reshaping a tensor via view."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.arange(12)
    m = t.view(3, 4)
    c = t.view(2, 2, 3)

    print(t.shape)
    print(m.shape)
    print(c.shape)
    return t, m, c


if __name__ == "__main__":
    main()
