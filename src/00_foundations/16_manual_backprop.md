# Exercise 16: Manual Backpropagation

Implement forward and backward passes for a tiny 2-layer MLP **without using
`autograd`**. This is the cleanest way to internalize what `loss.backward()`
actually does.

The network:

```
x  : (B, 2)   input
W1 : (2, 3)   first weights
b1 : (3,)     first bias
h  = ReLU(x @ W1 + b1)         (B, 3)
W2 : (3, 1)   second weights
b2 : (1,)     second bias
y_hat = h @ W2 + b2            (B, 1)
loss  = mean((y_hat - y) ** 2)  scalar
```

Hand-derive the gradients of the loss with respect to each parameter using the
chain rule, write them in code, and compare against PyTorch's autograd to
confirm they match.

## Tasks

- Fix a small batch with `torch.manual_seed(0)`: `B = 4`, `x = randn(4, 2)`,
  `y = randn(4, 1)`. Initialize `W1, b1, W2, b2` with `randn` of the right
  shapes (use `requires_grad=True` so you can compare against autograd later).
- Forward: compute `h`, `y_hat`, `loss`.
- Backward by hand:
  - `dL/dy_hat   = 2 (y_hat - y) / B`
  - `dL/dW2      = h.T @ dL/dy_hat`
  - `dL/db2      = sum(dL/dy_hat, axis=0)`
  - `dL/dh       = dL/dy_hat @ W2.T`
  - `dL/dz1      = dL/dh * (z1 > 0)`        where `z1 = x @ W1 + b1`
  - `dL/dW1      = x.T @ dL/dz1`
  - `dL/db1      = sum(dL/dz1, axis=0)`
- Call `loss.backward()` and compare each manual gradient against `W1.grad`,
  `b1.grad`, `W2.grad`, `b2.grad` using `torch.allclose`. They should match.

## Hints

- Keep your tensors detached during the manual backward; you only need autograd
  for the *verification* step at the end.
- ReLU's gradient is `1` where the pre-activation was positive, `0` otherwise.
