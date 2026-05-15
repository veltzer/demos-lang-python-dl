#!/usr/bin/env python

"""jax.grad takes the derivative of a function."""

import jax


def f(x: float) -> float:
    return x ** 3 + 2 * x


df = jax.grad(f)
print(df(3.0))
