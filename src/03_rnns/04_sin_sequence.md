# Exercise 04: Predict a Sine Wave

Generate `sin(t)` for `t` in `[0, 8π]` with 200 samples. Use sliding windows
of length 20 as inputs and the next value as target.

Train a `nn.LSTM(1, 16) -> nn.Linear(16, 1)` model with `MSELoss` and Adam.

Print the loss every 50 epochs over 300 epochs. It should drop substantially.
