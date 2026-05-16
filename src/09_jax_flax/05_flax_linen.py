#!/usr/bin/env python

"""A simple Flax linen.Module MLP."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)


def main() -> tuple[MLP, Any, Any]:
    key = jax.random.PRNGKey(0)
    model = MLP()
    params = model.init(key, jnp.ones((1, 10)))
    out = model.apply(params, jnp.ones((4, 10)))
    print(out.shape)  # type: ignore[union-attr]
    return model, params, out


if __name__ == "__main__":
    main()
