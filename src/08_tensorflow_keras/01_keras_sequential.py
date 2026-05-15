#!/usr/bin/env python

"""A simple Keras Sequential model."""

import keras

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.summary()
