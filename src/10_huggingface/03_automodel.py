#!/usr/bin/env python

"""Extract BERT embeddings for a sentence."""

import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()

inputs = tok("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
