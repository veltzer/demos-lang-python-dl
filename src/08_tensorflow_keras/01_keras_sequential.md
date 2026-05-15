# Exercise 01: Keras Sequential

Build:

```
keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax'),
])
```

Print `model.summary()`.
