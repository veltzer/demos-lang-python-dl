#!/usr/bin/env python

"""Vectorize a function over a leading axis with vmap."""

import jax
import jax.numpy as jnp


def dot(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(u, v)


batched = jax.vmap(dot)

key = jax.random.PRNGKey(0)
ka, kb = jax.random.split(key)
a = jax.random.normal(ka, (5, 3))
b = jax.random.normal(kb, (5, 3))
print(batched(a, b))
