#!/usr/bin/env python

"""Tests for src/05_rnn solutions."""

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


def test_rnn_shapes() -> None:
    mod = _load("src/05_rnn/00_rnn_shapes.py")
    out, h = mod.main()
    assert tuple(out.shape) == (3, 10, 8)
    assert tuple(h.shape) == (1, 3, 8)


def test_lstm_shapes() -> None:
    mod = _load("src/05_rnn/01_lstm_shapes.py")
    out, h, c = mod.main()
    assert tuple(out.shape) == (2, 7, 6)
    assert tuple(h.shape) == (2, 2, 6)
    assert tuple(c.shape) == (2, 2, 6)


def test_gru_shapes() -> None:
    mod = _load("src/05_rnn/02_gru_shapes.py")
    out, h = mod.main()
    # bidirectional doubles the hidden_size on output, and h has 2 layers (one per direction)
    assert tuple(out.shape) == (1, 6, 10)
    assert tuple(h.shape) == (2, 1, 5)


def test_manual_rnn_cell() -> None:
    mod = _load("src/05_rnn/03_manual_rnn_cell.py")
    h = mod.main()
    assert tuple(h.shape) == (1, 4)
    # hidden state should be bounded by tanh
    assert torch.all(h.abs() <= 1.0)


def test_sin_sequence() -> None:
    mod = _load("src/05_rnn/04_sin_sequence.py")
    _model, final_loss = mod.main()
    # LSTM should learn the sine forecast reasonably well after 300 epochs
    assert final_loss < 0.1


def test_packed_sequence() -> None:
    mod = _load("src/05_rnn/05_packed_sequence.py")
    out, lengths = mod.main()
    # 3 sequences, max length 5, hidden_size 8
    assert tuple(out.shape) == (3, 5, 8)
    assert sorted(lengths.tolist(), reverse=True) == [5, 4, 3]


def test_embedding() -> None:
    mod = _load("src/05_rnn/06_embedding.py")
    out = mod.main()
    assert tuple(out.shape) == (2, 3, 4)
