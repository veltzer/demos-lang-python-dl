#!/usr/bin/env python

"""Basic jax.numpy operations."""

import jax.numpy as jnp


def main() -> tuple[jnp.ndarray, jnp.ndarray]:
    a = jnp.array([1.0, 2.0, 3.0])
    doubled = a * 2
    total = a.sum()
    print(doubled)
    print(total)
    return doubled, total


if __name__ == "__main__":
    main()
