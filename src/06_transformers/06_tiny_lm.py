#!/usr/bin/env python

"""Tiny causal Transformer language model."""

import torch
from torch import nn

torch.manual_seed(0)

vocab = 50
dim = 32

emb = nn.Embedding(vocab, dim)
layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=2)
head = nn.Linear(dim, vocab)

ids = torch.randint(0, vocab, (2, 10))
x = emb(ids)
mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()
out = encoder(x, mask=mask)
logits = head(out)
print(logits.shape)
