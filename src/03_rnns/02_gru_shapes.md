# Exercise 02: GRU Shapes

`nn.GRU` has one hidden state, not two. Build
`nn.GRU(input_size=3, hidden_size=5, bidirectional=True, batch_first=True)`.

Pass a `(1, 6, 3)` input. Print output and `h_n` shapes — note the
bidirectional doubling.
