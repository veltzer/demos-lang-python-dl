# Exercise 03: jax.vmap

`jax.vmap` vectorizes a function over a leading axis.

Define `dot(a, b) = jnp.dot(a, b)` (operates on 1-D vectors). Use `vmap` to lift
it to operate on batches: `(N, D)` vs `(N, D)` and return `(N,)` element-wise
dot products.

Run it on two random `(5, 3)` arrays.
