#!/usr/bin/env python

"""train() vs eval() mode affects Dropout."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    x = torch.ones(1, 4)

    model.train()
    t1 = model(x)
    print(t1)
    t2 = model(x)
    print(t2)

    model.eval()
    e1 = model(x)
    print(e1)
    e2 = model(x)
    print(e2)
    return t1, t2, e1, e2


if __name__ == "__main__":
    main()
