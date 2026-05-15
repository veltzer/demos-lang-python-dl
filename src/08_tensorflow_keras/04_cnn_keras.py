#!/usr/bin/env python

"""Small Keras CNN."""

import keras

model = keras.Sequential([
    keras.layers.Conv2D(8, 3, activation="relu", padding="same",
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])

model.summary()
