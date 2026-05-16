#!/usr/bin/env python

"""Plain gradient descent without torch.optim."""

import torch


def main() -> float:
    x = torch.tensor(0.0, requires_grad=True)
    lr = 0.1

    for _ in range(50):
        loss = (x - 5) ** 2
        loss.backward()
        assert x.grad is not None
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()

    print(x.item())
    return x.item()


if __name__ == "__main__":
    main()
