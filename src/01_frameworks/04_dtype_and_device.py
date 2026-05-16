#!/usr/bin/env python

"""Tensor dtype conversion and device inspection."""

import torch


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    t64 = t.double()
    ti = t.to(torch.int32)

    print(t, t.dtype)
    print(t64, t64.dtype)
    print(ti, ti.dtype)
    print("device:", t.device)
    return t, t64, ti


if __name__ == "__main__":
    main()
