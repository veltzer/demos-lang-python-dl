#!/usr/bin/env python

"""Tests for src/06_transformers solutions."""

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def _load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scaled_dot_product() -> None:
    mod = _load("src/06_transformers/00_scaled_dot_product.py")
    out = mod.main()
    assert tuple(out.shape) == (2, 4, 8)


def test_multihead_attention() -> None:
    mod = _load("src/06_transformers/01_multihead_attention.py")
    out, weights = mod.main()
    assert tuple(out.shape) == (2, 6, 16)
    # averaged attention weights over heads: (batch, seq, seq)
    assert tuple(weights.shape) == (2, 6, 6)


def test_causal_mask() -> None:
    mod = _load("src/06_transformers/02_causal_mask.py")
    weights = mod.main()
    # weights shape: (1, 4, 4)
    assert tuple(weights.shape) == (1, 4, 4)
    w = weights.squeeze(0)
    # rows must sum to 1
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)
    # upper triangle (above diag) must be zero (causal)
    for i in range(4):
        for j in range(i + 1, 4):
            assert w[i, j].item() == 0.0


def test_positional_encoding() -> None:
    mod = _load("src/06_transformers/03_positional_encoding.py")
    pe = mod.main()
    assert tuple(pe.shape) == (20, 16)
    # position 0: sin(0) = 0 on even indices, cos(0) = 1 on odd indices
    assert torch.allclose(pe[0, 0::2], torch.zeros(8), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(8), atol=1e-6)


def test_encoder_layer() -> None:
    mod = _load("src/06_transformers/04_encoder_layer.py")
    out = mod.main()
    assert tuple(out.shape) == (3, 8, 16)


def test_encoder_stack() -> None:
    mod = _load("src/06_transformers/05_encoder_stack.py")
    out = mod.main()
    assert tuple(out.shape) == (3, 8, 16)


def test_tiny_lm() -> None:
    mod = _load("src/06_transformers/06_tiny_lm.py")
    logits = mod.main()
    # batch=2, seq=10, vocab=50
    assert tuple(logits.shape) == (2, 10, 50)
