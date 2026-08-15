#!/usr/bin/env python

"""Common activation functions side by side."""

import torch
from torch import nn


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    relu = nn.ReLU()(x)
    sigmoid = nn.Sigmoid()(x)
    tanh = nn.Tanh()(x)
    softmax = nn.Softmax(dim=-1)(x)

    print("relu:   ", relu)
    print("sigmoid:", sigmoid)
    print("tanh:   ", tanh)
    print("softmax:", softmax)
    return relu, sigmoid, tanh, softmax


if __name__ == "__main__":
    main()
