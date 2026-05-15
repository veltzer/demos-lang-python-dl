# Exercise 01: Gradient of a Sum

Let `x = [1.0, 2.0, 3.0]` with `requires_grad=True`. Compute `loss = (x**2).sum()`
and call `loss.backward()`.

Print `x.grad`. You should see `[2., 4., 6.]`.
