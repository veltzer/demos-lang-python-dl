# Exercise 06: Tiny Causal Language Model

Build a minimal causal LM that maps token IDs to next-token logits:

- `nn.Embedding(vocab=50, dim=32)`
- a single `TransformerEncoderLayer(d_model=32, nhead=4)` with a *causal mask*
- `nn.Linear(32, 50)` head

Pass a `(2, 10)` long tensor of token IDs in `[0, 50)` through it and print the
logits shape: `(2, 10, 50)`.
