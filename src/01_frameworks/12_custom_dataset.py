#!/usr/bin/env python

"""Custom Dataset subclass."""

from typing import override

import torch
from torch.utils.data import DataLoader, Dataset


class SquaresDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(index), torch.tensor(index * index)


def main() -> tuple[SquaresDataset, list[tuple[torch.Tensor, torch.Tensor]]]:
    ds = SquaresDataset(10)
    loader = DataLoader(ds, batch_size=2)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, batch in enumerate(loader):
        print(batch)
        batches.append(batch)
        if i == 4:
            break
    return ds, batches


if __name__ == "__main__":
    main()
