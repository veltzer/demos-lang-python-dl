#!/usr/bin/env python

"""Disabling gradient tracking."""

import torch


def main() -> tuple[bool, bool]:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    with torch.no_grad():
        y = x * 2
    print("no_grad:", y.requires_grad)

    z = x.detach() * 2
    print("detach:", z.requires_grad)
    return y.requires_grad, z.requires_grad


if __name__ == "__main__":
    main()
