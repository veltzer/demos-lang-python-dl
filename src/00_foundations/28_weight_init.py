#!/usr/bin/env python

"""Compare weight-initialization schemes."""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> dict[str, float]:
    results: dict[str, float] = {}
    initializers: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("normal(0, 0.01)", lambda w: nn.init.normal_(w, 0, 0.01)),
        ("xavier_normal", nn.init.xavier_normal_),
        ("kaiming_normal", lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")),
    ]
    for name, fn in initializers:
        layer = nn.Linear(100, 100)
        fn(layer.weight)
        std = layer.weight.std().item()
        results[name] = std
        print(f"{name:18s} std={std:.4f}")
    return results


if __name__ == "__main__":
    main()
