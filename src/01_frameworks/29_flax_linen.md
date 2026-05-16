# Exercise 05: A Flax Linen Model

Define a Flax `nn.Module` MLP with `Dense(32) -> ReLU -> Dense(3)`.

Initialize its parameters by calling `model.init(key, sample_input)` on a
sample `(1, 10)` input, then run `model.apply(params, x)` on a `(4, 10)` input
and print the output shape.
