#!/usr/bin/env python

"""Zero-shot classification with no fine-tuning."""

from transformers import pipeline

clf = pipeline("zero-shot-classification")
out = clf(
    "I am planning my next trip to Tokyo.",
    candidate_labels=["travel", "cooking", "politics"],
)
print(out)
