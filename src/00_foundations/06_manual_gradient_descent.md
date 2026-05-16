# Exercise 06: Manual Gradient Descent

Minimize `f(x) = (x - 5) ** 2` with raw autograd — no `torch.optim`.

- Start at `x = 0.0` with `requires_grad=True`.
- Step size `0.1`.
- Run 50 iterations: in each, compute the loss, call `backward()`, then
  update `x.data -= lr * x.grad` and zero `x.grad`.
- Print the final `x` — it should be very close to `5.0`.
