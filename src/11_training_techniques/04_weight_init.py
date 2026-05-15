#!/usr/bin/env python

"""Compare weight-initialization schemes."""

from torch import nn

for name, fn in [
    ("normal(0, 0.01)", lambda w: nn.init.normal_(w, 0, 0.01)),
    ("xavier_normal", nn.init.xavier_normal_),
    ("kaiming_normal", lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")),
]:
    layer = nn.Linear(100, 100)
    fn(layer.weight)
    print(f"{name:18s} std={layer.weight.std().item():.4f}")
