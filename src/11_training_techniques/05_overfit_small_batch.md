# Exercise 05: Overfit a Single Batch

A standard sanity check: can your model overfit a tiny batch of data?

Generate `x = (8, 4)` random, `y = (8,)` random class labels (3 classes). Build
an MLP `4 -> 64 -> 3`. Train it for 500 steps on just this batch with Adam.
Print the loss every 100 steps. It should drop to nearly zero.
