# Exercise 00: RNN Input/Output Shapes

Create `nn.RNN(input_size=5, hidden_size=8, batch_first=True)`.

Push a `(3, 10, 5)` input — batch of 3 sequences, length 10, 5 features per
timestep — and print:

- the output (per-timestep hidden states) shape: `(3, 10, 8)`
- the final hidden state shape: `(1, 3, 8)`
