#!/usr/bin/env python

"""Save and load a model's state_dict."""

import tempfile
import torch
from torch import nn


def main() -> bool:
    torch.manual_seed(0)
    m1 = nn.Linear(4, 2)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(m1.state_dict(), path)

    m2 = nn.Linear(4, 2)
    m2.load_state_dict(torch.load(path))

    x = torch.randn(1, 4)
    equal = torch.allclose(m1(x), m2(x))
    print(equal)
    return equal


if __name__ == "__main__":
    main()
