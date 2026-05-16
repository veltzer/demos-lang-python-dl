# Exercise 00: Scaled Dot-Product Attention

Implement the attention formula by hand:

```
attn(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Given `Q, K, V` of shape `(2, 4, 8)` (batch=2, seq_len=4, d_k=8), compute the
output. Print its shape — should be `(2, 4, 8)`.
