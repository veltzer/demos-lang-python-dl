# Exercise 01: Adam on a 2-D Quadratic

Minimize `f(x, y) = (x - 3) ** 2 + (y + 4) ** 2` using `torch.optim.Adam`.

- Wrap `(x, y)` in a parameter tensor `p`.
- Use `Adam([p], lr=0.1)`.
- 200 iterations.
- Print final `p` (should be approximately `[3.0, -4.0]`).
