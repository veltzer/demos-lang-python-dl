# Exercise 00: requires_grad

Create a tensor `x = 3.0` with `requires_grad=True`. Compute `y = x ** 2`,
call `y.backward()`, and print `x.grad`.

The derivative of `x^2` at `x = 3` is `6`, so you should see `tensor(6.)`.
