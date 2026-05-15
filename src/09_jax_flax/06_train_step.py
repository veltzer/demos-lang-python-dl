#!/usr/bin/env python

"""Train a Flax model with raw jax.grad + manual SGD."""

import jax
import jax.numpy as jnp
from flax import linen as nn


class Linear(nn.Module):
    @nn.compact
    def __call__(self, inp: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(1)(inp)


key = jax.random.PRNGKey(0)
x_key, init_key = jax.random.split(key)

x = jax.random.uniform(x_key, (100, 1)) * 10
y = 2 * x + 1

model = Linear()
params = model.init(init_key, x)


def loss_fn(p, inp, target):
    pred = model.apply(p, inp)
    return jnp.mean((pred - target) ** 2)


grad_fn = jax.jit(jax.grad(loss_fn))
lr = 0.01
for _ in range(500):
    grads = grad_fn(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

print(loss_fn(params, x, y))
