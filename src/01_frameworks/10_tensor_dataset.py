#!/usr/bin/env python

"""TensorDataset wraps parallel tensors."""

import torch
from torch.utils.data import TensorDataset


def main() -> TensorDataset:
    torch.manual_seed(0)

    x = torch.randn(100, 4)
    y = torch.randint(0, 3, (100,))

    ds = TensorDataset(x, y)
    print(len(ds))
    print(ds[0])
    return ds


if __name__ == "__main__":
    main()
