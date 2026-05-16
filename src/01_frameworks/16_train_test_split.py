#!/usr/bin/env python

"""Split a dataset with random_split."""

import torch
from torch.utils.data import TensorDataset, Subset, random_split


def main() -> tuple[Subset, Subset]:
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(1000, 4), torch.randint(0, 3, (1000,)))
    train, test = random_split(ds, [800, 200])
    print(len(train), len(test))
    return train, test


if __name__ == "__main__":
    main()
