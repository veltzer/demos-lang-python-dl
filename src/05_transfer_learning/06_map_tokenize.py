#!/usr/bin/env python

"""Tokenize a dataset via Dataset.map."""

from typing import TYPE_CHECKING

from datasets import load_dataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from datasets.dataset_dict import DatasetDict


def main() -> tuple[Dataset | DatasetDict, list[str]]:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tok is None:
        raise RuntimeError("could not load the tokenizer")
    ds = load_dataset("glue", "sst2", split="train").select(range(100))

    def tokenize(batch: dict[str, list[str]]):
        return tok(batch["sentence"], padding="max_length", truncation=True, max_length=32)

    ds = ds.map(tokenize, batched=True)
    keys = list(ds.column_names)
    print(keys)
    return ds, keys


if __name__ == "__main__":
    main()
