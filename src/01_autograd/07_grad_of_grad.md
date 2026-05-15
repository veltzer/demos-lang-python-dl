# Exercise 07: Second-Order Gradients

Compute the second derivative of `f(x) = x ** 3` at `x = 2`.

Use `torch.autograd.grad(..., create_graph=True)` on the first derivative, then
differentiate it again.

The expected answer: `f''(x) = 6x`, so at `x = 2` the result is `12`.
