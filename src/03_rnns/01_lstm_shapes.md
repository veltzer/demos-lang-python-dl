# Exercise 01: LSTM Returns (h, c)

Build `nn.LSTM(4, 6, num_layers=2, batch_first=True)` and pass a `(2, 7, 4)`
batch through it.

Print:
- output shape: `(2, 7, 6)`
- `h_n` shape: `(2, 2, 6)` — layers × batch × hidden
- `c_n` shape: `(2, 2, 6)`
