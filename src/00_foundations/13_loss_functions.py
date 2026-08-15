#!/usr/bin/env python

"""Three canonical losses."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = nn.MSELoss()
    mse_val = mse(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5, 3.5]))
    print(mse_val)

    bce = nn.BCEWithLogitsLoss()
    bce_val = bce(torch.tensor([1.0, -1.0]), torch.tensor([1.0, 0.0]))
    print(bce_val)

    ce = nn.CrossEntropyLoss()
    ce_val = ce(torch.tensor([[2.0, 0.5, 1.0]]), torch.tensor([0]))
    print(ce_val)
    return mse_val, bce_val, ce_val


if __name__ == "__main__":
    main()
