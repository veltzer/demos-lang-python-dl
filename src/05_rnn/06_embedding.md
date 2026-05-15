# Exercise 06: Embedding Layer

Create an `nn.Embedding(num_embeddings=20, embedding_dim=4)`. Pass a tensor of
token IDs `[[1, 5, 17], [3, 8, 0]]` through it and print the resulting shape —
expect `(2, 3, 4)`.

This is the canonical first layer of any text model.
