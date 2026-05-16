#!/usr/bin/env python

"""NumPy <-> PyTorch zero-copy bridge."""

import numpy as np
import torch


def main() -> tuple[np.ndarray, torch.Tensor]:
    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)

    t.mul_(2)

    print("numpy:", arr)
    print("tensor:", t)
    print("back to numpy:", t.numpy())
    return arr, t


if __name__ == "__main__":
    main()
