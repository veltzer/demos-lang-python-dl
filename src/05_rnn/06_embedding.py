#!/usr/bin/env python

"""Token embedding lookup."""

import torch
from torch import nn


def main() -> torch.Tensor:
    emb = nn.Embedding(num_embeddings=20, embedding_dim=4)
    ids = torch.tensor([[1, 5, 17], [3, 8, 0]])
    out = emb(ids)
    print(out.shape)
    return out


if __name__ == "__main__":
    main()
