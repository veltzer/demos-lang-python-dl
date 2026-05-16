#!/usr/bin/env python

"""Zero-shot classification with no fine-tuning."""

from transformers import pipeline


def main() -> dict:
    clf = pipeline("zero-shot-classification")
    out = clf(
        "I am planning my next trip to Tokyo.",
        candidate_labels=["travel", "cooking", "politics"],
    )
    print(out)
    return out


if __name__ == "__main__":
    main()
