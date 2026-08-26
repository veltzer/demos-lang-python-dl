#!/usr/bin/env python

"""Create a basic tf.constant tensor."""

# tensorflow has no cp314 wheel yet, and CI runs Python 3.14, so it is not
# installed there. Guard the import so this file still imports cleanly; the
# test skips when tf is missing.
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised only where tf is absent
    tf = None


def main():
    t = tf.constant([1, 2, 3, 4, 5])
    print(t)
    print(t.dtype)
    print(t.shape)
    return t


if __name__ == "__main__":
    main()
