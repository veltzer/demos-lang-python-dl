# Exercise 04: Count Parameters

Given a model, count its total number of trainable parameters.

Define an MLP with hidden sizes `100 -> 50 -> 10` from a `20`-dim input. Iterate
`model.parameters()`, sum each tensor's `.numel()` filtered by
`requires_grad`, and print the total.

Verify by hand: a `Linear(20, 100)` has `20*100 + 100 = 2100` params, etc.
