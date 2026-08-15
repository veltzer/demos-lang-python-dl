#!/usr/bin/env python

"""Continue a prompt with a small causal LM."""

from transformers import pipeline


def main() -> list[dict]:
    gen = pipeline("text-generation", model="distilgpt2")
    out = gen("In the future, AI will", max_new_tokens=30)
    print(out)
    return out


if __name__ == "__main__":
    main()
