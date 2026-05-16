# Exercise 02: Chain Rule

Let `x` be a scalar tensor with `requires_grad=True` and `x = 2.0`. Compute:

```
y = x * 3
z = y * y + y
```

Call `z.backward()` and print `x.grad`.

Verify by hand: `z = (3x)^2 + 3x`, so `dz/dx = 18x + 3 = 39` at `x = 2`.
