"""Tests for src/10_huggingface/* solutions.

These exercises hit the HuggingFace hub. Every test is marked 'network'
and is skipped unless pytest is run with `-m network`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.network


def test_00_sentiment_pipeline():
    mod = load("src/05_transfer_learning/00_sentiment_pipeline.py")
    results = mod.main()
    assert len(results) == 2
    for _text, out in results:
        assert isinstance(out, list)
        assert "label" in out[0]


def test_01_zero_shot():
    mod = load("src/05_transfer_learning/01_zero_shot.py")
    out = mod.main()
    assert "labels" in out
    assert "scores" in out
    assert set(out["labels"]) == {"travel", "cooking", "politics"}


def test_02_tokenizer():
    mod = load("src/05_transfer_learning/02_tokenizer.py")
    ids, tokens = mod.main()
    assert len(ids) == len(tokens)
    assert tokens[0] == "[CLS]"
    assert tokens[-1] == "[SEP]"


def test_03_automodel():
    mod = load("src/05_transfer_learning/03_automodel.py")
    h = mod.main()
    assert h.shape[0] == 1
    assert h.shape[-1] == 768  # bert-base hidden size


def test_04_text_generation():
    mod = load("src/05_transfer_learning/04_text_generation.py")
    out = mod.main()
    assert isinstance(out, list) and len(out) >= 1
    assert "generated_text" in out[0]


def test_05_datasets_basic():
    mod = load("src/05_transfer_learning/05_datasets_basic.py")
    ds = mod.main()
    assert "train" in ds


def test_06_map_tokenize():
    mod = load("src/05_transfer_learning/06_map_tokenize.py")
    _ds, keys = mod.main()
    assert "input_ids" in keys
