# Exercise 03: no_grad Context

Show two ways to compute a tensor expression without tracking gradients:

1. Wrap the computation in a `with torch.no_grad():` block.
2. Call `.detach()` on the input.

In both cases, the result tensor should have `requires_grad == False`.
