#!/usr/bin/env python

"""Output and hidden-state shapes of nn.RNN."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    rnn = nn.RNN(input_size=5, hidden_size=8, batch_first=True)
    x = torch.randn(3, 10, 5)
    out, h = rnn(x)
    print(out.shape)
    print(h.shape)
    return out, h


if __name__ == "__main__":
    main()
