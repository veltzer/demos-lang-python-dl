#!/usr/bin/env python

"""Save and reload a Keras model end-to-end."""

import os
import tempfile
import numpy as np
import keras


def main() -> bool:
    m1 = keras.Sequential([keras.layers.Dense(4, input_shape=(3,))])
    m1.compile(optimizer="adam", loss="mse")

    x = np.random.randn(2, 3).astype("float32")
    y1 = m1.predict(x, verbose=0)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        path = f.name
    m1.save(path)

    m2 = keras.models.load_model(path)
    y2 = m2.predict(x, verbose=0)

    same = bool(np.allclose(y1, y2))
    print(same)
    os.unlink(path)
    return same


if __name__ == "__main__":
    main()
