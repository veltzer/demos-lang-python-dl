#!/usr/bin/env python

"""Load a small HuggingFace dataset."""

from datasets import load_dataset

ds = load_dataset("glue", "sst2")
for split in ds:
    print(split, len(ds[split]))
print(ds["train"][0])
