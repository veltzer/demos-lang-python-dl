#!/usr/bin/env python

"""Compute a gradient with tf.GradientTape."""

import tensorflow as tf


def main() -> tf.Tensor:
    x = tf.Variable(4.0)
    with tf.GradientTape() as tape:
        y = x ** 3

    grad = tape.gradient(y, x)
    print(grad)
    return grad


if __name__ == "__main__":
    main()
