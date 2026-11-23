#!/usr/bin/env python

"""Save and reload a Keras model end-to-end."""

import os
import tempfile

import keras
import numpy as np


def main() -> bool:
    m1 = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(4)])
    m1.compile(optimizer="adam", loss="mse")

    x = np.random.default_rng().standard_normal((2, 3)).astype("float32")
    # one training step builds the optimizer's state, which is saved with it
    m1.fit(x, np.zeros((2, 4), dtype="float32"), epochs=1, verbose=0)
    y1 = m1.predict(x, verbose=0)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        path = f.name
    m1.save(path)

    m2 = keras.models.load_model(path)
    y2 = m2.predict(x, verbose=0)

    same = np.allclose(y1, y2)
    print(same)
    os.unlink(path)
    return same


if __name__ == "__main__":
    main()
