# Exercise 03: Keras Functional API

The Functional API lets you build models that aren't simple stacks (multiple
inputs, branches, etc.).

Build a model with two inputs of shape `(8,)` and `(4,)` that:
- runs each through a Dense layer
- concatenates the results
- ends with a Dense(1) head.

Print `model.summary()`.
