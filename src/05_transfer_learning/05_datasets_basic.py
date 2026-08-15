#!/usr/bin/env python

"""Load a small HuggingFace dataset."""

from datasets import load_dataset


def main():
    ds = load_dataset("glue", "sst2")
    for split in ds:
        print(split, len(ds[split]))
    print(ds["train"][0])
    return ds


if __name__ == "__main__":
    main()
