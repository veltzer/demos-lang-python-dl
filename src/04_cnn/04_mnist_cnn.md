# Exercise 04: A Tiny CNN for MNIST-Shaped Inputs

You don't have MNIST loaded here. Just build a small CNN sized for the MNIST
input shape `(N, 1, 28, 28)` and `10` output classes:

```
Conv2d(1, 8, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(8, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(16*7*7, 10)
```

Push a random `(4, 1, 28, 28)` batch through it and print the output shape.
