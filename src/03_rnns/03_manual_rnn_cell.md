# Exercise 03: Roll Your Own RNN Cell

Implement the RNN recurrence by hand:

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
```

Use weight matrices of shape `(hidden, input)` and `(hidden, hidden)`. Loop over
the time dimension of a `(1, 5, 3)` input and print the final hidden state.
