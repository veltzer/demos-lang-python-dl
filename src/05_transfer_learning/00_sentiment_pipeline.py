#!/usr/bin/env python

"""Use a pre-trained sentiment-analysis (text-classification) pipeline."""

from typing import Any

from transformers import pipeline


def main() -> list[tuple[str, Any]]:
    clf = pipeline("text-classification")
    results = []
    for text in ("I love this movie.", "This is the worst day ever."):
        out = clf(text)
        print(text, "->", out)
        results.append((text, out))
    return results


if __name__ == "__main__":
    main()
