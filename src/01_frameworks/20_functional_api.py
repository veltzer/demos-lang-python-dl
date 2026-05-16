#!/usr/bin/env python

"""Two-input model via the Keras Functional API."""

import keras


def main() -> keras.Model:
    a = keras.Input(shape=(8,), name="a")
    b = keras.Input(shape=(4,), name="b")

    ha = keras.layers.Dense(16, activation="relu")(a)
    hb = keras.layers.Dense(16, activation="relu")(b)
    h = keras.layers.Concatenate()([ha, hb])
    out = keras.layers.Dense(1)(h)

    model = keras.Model(inputs=[a, b], outputs=out)
    model.summary()
    return model


if __name__ == "__main__":
    main()
