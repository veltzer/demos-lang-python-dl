#!/usr/bin/env python

"""Use a pre-trained sentiment-analysis pipeline."""

from typing import Any

from transformers import pipeline


def main() -> list[tuple[str, Any]]:
    clf = pipeline("sentiment-analysis")  # type: ignore[call-overload]
    results = []
    for text in ("I love this movie.", "This is the worst day ever."):
        out = clf(text)
        print(text, "->", out)
        results.append((text, out))
    return results


if __name__ == "__main__":
    main()
