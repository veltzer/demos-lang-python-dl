#!/usr/bin/env python

"""A simple Keras Sequential model."""

import keras


def main() -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])

    model.summary()
    return model


if __name__ == "__main__":
    main()
