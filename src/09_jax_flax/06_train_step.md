# Exercise 06: Flax + JAX Training Step

Train a Flax MLP to fit `y = 2x + 1` on 100 points.

- Define a one-layer MLP `Dense(1)`.
- Loss is MSE.
- Use `jax.grad` to compute gradients of the loss w.r.t. parameters.
- Update with plain SGD (`params - lr * grads`) using `jax.tree_util.tree_map`.

Run 500 steps and print the final loss.
