# Exercise 04: PRNG Keys

JAX uses explicit, splittable PRNG keys instead of a global state.

Create a key from `jax.random.PRNGKey(42)`, split it into 3, and use each one
to draw a `(2, 2)` standard normal sample. Print all three.
