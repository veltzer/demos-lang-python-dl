# Exercise 03: Custom nn.Module

Define your own `nn.Module` subclass `TwoLayerNet(in_dim, hidden, out_dim)`:

- `__init__` saves two `nn.Linear` layers.
- `forward(x)` runs `linear1 -> relu -> linear2` and returns the output.

Instantiate with `(20, 64, 3)`, push a `(5, 20)` input through it, print the
output shape.
