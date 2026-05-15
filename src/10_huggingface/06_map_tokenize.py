#!/usr/bin/env python

"""Tokenize a dataset via Dataset.map."""

from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = load_dataset("glue", "sst2", split="train").select(range(100))


def tokenize(batch):
    return tok(batch["sentence"], padding="max_length", truncation=True, max_length=32)


ds = ds.map(tokenize, batched=True)
print(list(ds[0].keys()))
