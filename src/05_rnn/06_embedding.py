#!/usr/bin/env python

"""Token embedding lookup."""

import torch
from torch import nn

emb = nn.Embedding(num_embeddings=20, embedding_dim=4)
ids = torch.tensor([[1, 5, 17], [3, 8, 0]])
print(emb(ids).shape)
