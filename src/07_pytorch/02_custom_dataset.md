# Exercise 02: Custom Dataset

Build your own `Dataset` subclass `SquaresDataset(n)` that:

- holds `n` integers `0..n-1`.
- `__len__` returns `n`.
- `__getitem__(i)` returns `(i, i*i)` as a `(int, int)` tuple of tensors.

Wrap it in a DataLoader and print the first 5 items.
