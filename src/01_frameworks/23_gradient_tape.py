#!/usr/bin/env python

"""Compute a gradient with tf.GradientTape."""

# tensorflow has no cp314 wheel yet, and CI runs Python 3.14, so it is not
# installed there. Guard the import so this file still imports cleanly; the
# test skips when tf is missing.
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised only where tf is absent
    tf = None


def main():
    x = tf.Variable(4.0)
    with tf.GradientTape() as tape:
        y = x ** 3

    grad = tape.gradient(y, x)
    print(grad)
    return grad


if __name__ == "__main__":
    main()
