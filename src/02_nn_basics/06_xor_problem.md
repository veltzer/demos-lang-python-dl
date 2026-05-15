# Exercise 06: Learn the XOR Function

Train a small MLP (`2 -> 4 -> 1` with `Tanh` activations) to solve XOR:

```
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0],   [1],   [1],   [0]]
```

Use `BCEWithLogitsLoss` and plain `SGD(lr=0.5)`. Run a few thousand iterations.
Print the model's predictions on the four inputs after training — they should
match `Y` after rounding.
