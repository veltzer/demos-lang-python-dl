#!/usr/bin/env python

"""Zero-shot classification with no fine-tuning."""

from typing import Any

from transformers import pipeline


def main() -> dict[str, Any]:
    clf = pipeline("zero-shot-classification")
    out: dict[str, Any] = clf(
        "I am planning my next trip to Tokyo.",
        candidate_labels=["travel", "cooking", "politics"],
    )
    print(out)
    return out


if __name__ == "__main__":
    main()
