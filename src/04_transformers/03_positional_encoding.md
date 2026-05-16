# Exercise 03: Sinusoidal Positional Encoding

Implement the original Transformer positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Generate a `(max_len=20, d=16)` table and print its shape and the first row.
