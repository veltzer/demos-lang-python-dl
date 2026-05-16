# Exercise 02: Causal Mask

For autoregressive models, position `i` may only attend to positions `0..i`.

Build a `(4, 4)` boolean upper-triangular mask (True for "block") and apply it
to attention scores by setting masked entries to `-inf` before the softmax.

Print the resulting attention weights — they should be lower-triangular.
