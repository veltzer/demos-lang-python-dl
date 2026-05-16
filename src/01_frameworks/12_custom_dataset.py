#!/usr/bin/env python

"""Custom Dataset subclass."""

import torch
from torch.utils.data import Dataset, DataLoader


class SquaresDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(idx), torch.tensor(idx * idx)


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
