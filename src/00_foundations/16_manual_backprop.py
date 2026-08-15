#!/usr/bin/env python

"""Manual forward+backward through a 2-layer MLP, verified against autograd."""

import torch


def main() -> tuple[bool, bool, bool, bool]:
    torch.manual_seed(0)

    b = 4
    x = torch.randn(b, 2)
    y = torch.randn(b, 1)

    w1 = torch.randn(2, 3, requires_grad=True)
    b1 = torch.randn(3, requires_grad=True)
    w2 = torch.randn(3, 1, requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)

    z1 = x @ w1 + b1
    h = torch.relu(z1)
    y_hat = h @ w2 + b2
    loss = ((y_hat - y) ** 2).mean()

    dl_dyhat = 2 * (y_hat - y) / b
    dl_dw2 = h.detach().T @ dl_dyhat.detach()
    dl_db2 = dl_dyhat.detach().sum(dim=0)
    dl_dh = dl_dyhat.detach() @ w2.detach().T
    dl_dz1 = dl_dh * (z1.detach() > 0).float()
    dl_dw1 = x.T @ dl_dz1
    dl_db1 = dl_dz1.sum(dim=0)

    loss.backward()
    assert w1.grad is not None and b1.grad is not None
    assert w2.grad is not None and b2.grad is not None

    ok_w1 = torch.allclose(dl_dw1, w1.grad, atol=1e-6)
    ok_b1 = torch.allclose(dl_db1, b1.grad, atol=1e-6)
    ok_w2 = torch.allclose(dl_dw2, w2.grad, atol=1e-6)
    ok_b2 = torch.allclose(dl_db2, b2.grad, atol=1e-6)

    print("W1 match:", ok_w1)
    print("b1 match:", ok_b1)
    print("W2 match:", ok_w2)
    print("b2 match:", ok_b2)
    return ok_w1, ok_b1, ok_w2, ok_b2


if __name__ == "__main__":
    main()
