#!/usr/bin/env python

"""JIT-compile a function with jax.jit."""

import jax
import jax.numpy as jnp


@jax.jit
def f(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.sin(x) ** 2)


print(f(jnp.arange(1000.0)))
