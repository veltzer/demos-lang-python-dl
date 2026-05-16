# Exercise 02: jax.jit

JIT-compile a function for speed.

Define `f(x) = jnp.sum(jnp.sin(x) ** 2)`. Wrap it with `jax.jit`. Call it on
`jnp.arange(1000.0)` once and print the result.
