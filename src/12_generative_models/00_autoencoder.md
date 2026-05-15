# Exercise 00: Linear Autoencoder

Build a tiny autoencoder for `(N, 8)` inputs:

```
Encoder: Linear(8 -> 2)
Decoder: Linear(2 -> 8)
```

Train it on random data so it learns the identity through the 2-D bottleneck.
Use MSE and Adam, 1000 steps. Print the final loss.
