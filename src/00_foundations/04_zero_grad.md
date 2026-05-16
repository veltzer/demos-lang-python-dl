# Exercise 04: Why zero_grad?

PyTorch accumulates gradients across backward calls. Demonstrate this:

- Create `x = 2.0` with `requires_grad=True`.
- Compute `y = x**2`, call `y.backward()`, print `x.grad` — expect `4`.
- Compute `y = x**2` again, call `y.backward()` again, print `x.grad` — expect `8`
  because the new gradient was *added* to the previous one.
- Now zero the gradient with `x.grad.zero_()` and repeat — back to `4`.
