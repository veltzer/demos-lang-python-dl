# Exercise 04: Simple Affine Flow

Define a single affine "flow" that maps `z ~ N(0, 1)` to `x = exp(s) * z + t`
with learnable scalars `s, t`.

Train (maximum likelihood) so that the model puts mass on a target distribution
of `N(3, 2)`. The change-of-variables formula says:

```
log p_X(x) = log p_Z(z) - s
```

where `z = (x - t) / exp(s)`.

Train for 1000 steps on samples from the target and print learned `s, t`.
Expected: `s ≈ log(2) ≈ 0.69`, `t ≈ 3`.
