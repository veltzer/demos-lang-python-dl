#!/usr/bin/env python

"""BERT tokenizer round-trip."""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok("Deep learning is fun.")
print(enc.input_ids)
print(tok.convert_ids_to_tokens(enc.input_ids))
