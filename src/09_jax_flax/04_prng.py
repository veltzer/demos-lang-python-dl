#!/usr/bin/env python

"""Splittable PRNG keys in JAX."""

import jax

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

print(jax.random.normal(k1, (2, 2)))
print(jax.random.normal(k2, (2, 2)))
print(jax.random.normal(k3, (2, 2)))
