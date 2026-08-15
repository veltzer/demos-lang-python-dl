#!/usr/bin/env python

"""Splittable PRNG keys in JAX."""

import jax
import jax.numpy as jnp


def main() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    s1 = jax.random.normal(k1, (2, 2))
    s2 = jax.random.normal(k2, (2, 2))
    s3 = jax.random.normal(k3, (2, 2))
    print(s1)
    print(s2)
    print(s3)
    return s1, s2, s3


if __name__ == "__main__":
    main()
