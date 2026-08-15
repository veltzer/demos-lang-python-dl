#!/usr/bin/env python

"""Indexing and slicing a 5x5 tensor."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.arange(25).view(5, 5)

    row = t[1]
    col = t[:, -1]
    sub = t[1:4, 1:4]

    print(row)
    print(col)
    print(sub)
    return row, col, sub


if __name__ == "__main__":
    main()
