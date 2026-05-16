#!/usr/bin/env python

"""compile() then fit() a Keras classifier on random data."""

import numpy as np
import keras


def main() -> tuple[keras.Model, "keras.callbacks.History"]:
    np.random.seed(0)

    x = np.random.randn(200, 10).astype("float32")
    y = np.random.randint(0, 3, size=(200,))

    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(10,)),
        keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(x, y, epochs=5, batch_size=32, verbose=2)
    return model, history


if __name__ == "__main__":
    main()
