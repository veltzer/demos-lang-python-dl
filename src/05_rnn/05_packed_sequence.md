# Exercise 05: Packed Sequences

When a batch has sequences of *different* lengths, you should use
`nn.utils.rnn.pack_padded_sequence` so the RNN ignores the padding.

Build three sequences of lengths 5, 3, 4 (with dim=2 features). Pad with
`pad_sequence`, pack them, run an `nn.LSTM(2, 8)`, then unpack with
`pad_packed_sequence`. Print the resulting output shape and per-sequence
lengths.
