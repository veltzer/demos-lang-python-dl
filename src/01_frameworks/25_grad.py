#!/usr/bin/env python

"""jax.grad takes the derivative of a function."""

import jax
import jax.numpy as jnp


def f(x: float) -> float:
    return x ** 3 + 2 * x


def main() -> jnp.ndarray:
    df = jax.grad(f)
    result = df(3.0)
    print(result)
    return result


if __name__ == "__main__":
    main()
