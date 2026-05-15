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


loader = DataLoader(SquaresDataset(10), batch_size=2)
for i, batch in enumerate(loader):
    print(batch)
    if i == 4:
        break
