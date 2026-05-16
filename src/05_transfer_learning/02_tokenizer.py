#!/usr/bin/env python

"""BERT tokenizer round-trip."""

from typing import Any

from transformers import AutoTokenizer


def main() -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tok("Deep learning is fun.")
    tokens = tok.convert_ids_to_tokens(enc.input_ids)
    print(enc.input_ids)
    print(tokens)
    return enc.input_ids, tokens


if __name__ == "__main__":
    main()
