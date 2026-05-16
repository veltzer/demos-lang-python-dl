#!/usr/bin/env python

"""Count trainable parameters of a model."""

from torch import nn


def main() -> int:
    model = nn.Sequential(
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total)
    return total


if __name__ == "__main__":
    main()
