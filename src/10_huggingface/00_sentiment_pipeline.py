#!/usr/bin/env python

"""Use a pre-trained sentiment-analysis pipeline."""

from transformers import pipeline

clf = pipeline("sentiment-analysis")
for text in ("I love this movie.", "This is the worst day ever."):
    print(text, "->", clf(text))
