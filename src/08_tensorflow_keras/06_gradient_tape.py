#!/usr/bin/env python

"""Compute a gradient with tf.GradientTape."""

import tensorflow as tf

x = tf.Variable(4.0)
with tf.GradientTape() as tape:
    y = x ** 3

print(tape.gradient(y, x))
