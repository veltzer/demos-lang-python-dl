#!/usr/bin/env python

"""Compare CE loss with and without label smoothing."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    target = torch.tensor([0])

    plain = nn.CrossEntropyLoss()(logits, target)
    smooth = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
    print(plain)
    print(smooth)
    return plain, smooth


if __name__ == "__main__":
    main()
