#!/usr/bin/env python

"""Manual RNN recurrence."""

import torch


def main() -> torch.Tensor:
    torch.manual_seed(0)

    input_size = 3
    hidden_size = 4

    w_xh = torch.randn(hidden_size, input_size)
    w_hh = torch.randn(hidden_size, hidden_size)
    b = torch.zeros(hidden_size)

    x = torch.randn(1, 5, input_size)
    h = torch.zeros(1, hidden_size)
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ w_xh.T + h @ w_hh.T + b)
    print(h)
    return h


if __name__ == "__main__":
    main()
