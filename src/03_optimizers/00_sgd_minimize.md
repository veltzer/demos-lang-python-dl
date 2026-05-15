# Exercise 00: SGD on a Quadratic

Use `torch.optim.SGD` to minimize `f(x) = (x - 7) ** 2`.

- Start `x = 0.0` with `requires_grad=True`.
- Build an `SGD([x], lr=0.1)` optimizer.
- Run 100 steps: zero gradients, compute loss, backward, step.
- Print the final `x`.
