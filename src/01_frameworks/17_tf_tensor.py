#!/usr/bin/env python

"""Create a basic tf.constant tensor."""

import tensorflow as tf


def main() -> tf.Tensor:
    t = tf.constant([1, 2, 3, 4, 5])
    print(t)
    print(t.dtype)
    print(t.shape)
    return t


if __name__ == "__main__":
    main()
