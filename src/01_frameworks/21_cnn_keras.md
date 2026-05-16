# Exercise 04: A CNN in Keras

Build a small CNN for `(28, 28, 1)` inputs:

```
Conv2D(8, 3, activation='relu', padding='same') -> MaxPool2D()
Conv2D(16, 3, activation='relu', padding='same') -> MaxPool2D()
Flatten -> Dense(10, activation='softmax')
```

Print `model.summary()`.
