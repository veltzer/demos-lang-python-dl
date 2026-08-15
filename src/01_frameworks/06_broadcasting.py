#!/usr/bin/env python

"""Broadcasting a row vector across a matrix."""

import torch


def main() -> torch.Tensor:
    m = torch.arange(12).view(3, 4)
    v = torch.arange(4) * 10

    out = m + v
    print(m)
    print(v)
    print(out)
    return out


if __name__ == "__main__":
    main()
