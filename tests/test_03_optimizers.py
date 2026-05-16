#!/usr/bin/env python

"""Tests for src/03_optimizers solutions."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType


def _load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sgd_minimize() -> None:
    mod = _load("src/03_optimizers/00_sgd_minimize.py")
    val = mod.main()
    # Should converge close to 7
    assert abs(val - 7.0) < 1e-3


def test_adam_minimize() -> None:
    mod = _load("src/03_optimizers/01_adam_minimize.py")
    p = mod.main()
    # Should converge near (3, -4)
    assert abs(p[0].item() - 3.0) < 0.05
    assert abs(p[1].item() + 4.0) < 0.05


def test_momentum() -> None:
    mod = _load("src/03_optimizers/02_momentum.py")
    plain, momentum = mod.main()
    assert not math.isnan(plain)
    assert not math.isnan(momentum)


def test_lr_scheduler() -> None:
    mod = _load("src/03_optimizers/03_lr_scheduler.py")
    lrs = mod.main()
    assert len(lrs) == 20
    # lr starts 0.1, halves every 5 steps
    # After 5 steps -> 0.05, after 10 -> 0.025, after 15 -> 0.0125, after 20 -> 0.00625
    assert abs(lrs[0] - 0.1) < 1e-9
    assert abs(lrs[4] - 0.05) < 1e-9
    assert abs(lrs[9] - 0.025) < 1e-9
    assert abs(lrs[14] - 0.0125) < 1e-9
    assert abs(lrs[19] - 0.00625) < 1e-9


def test_linear_regression() -> None:
    mod = _load("src/03_optimizers/04_linear_regression.py")
    w, b = mod.main()
    # True w=2, b=1; with noise we allow some tolerance
    assert abs(w - 2.0) < 0.1
    assert abs(b - 1.0) < 0.3


def test_minibatch() -> None:
    mod = _load("src/03_optimizers/05_minibatch.py")
    w, b = mod.main()
    # True w=2, b=1; minibatch should still get reasonably close
    assert abs(w - 2.0) < 0.2
    assert abs(b - 1.0) < 0.5


def test_weight_decay() -> None:
    mod = _load("src/03_optimizers/06_weight_decay.py")
    no_decay, with_decay = mod.main()
    # Both should still be in a reasonable neighbourhood of the true slope (2.0).
    # weight_decay biases toward zero but with this data/seed both end up near 2.
    assert abs(no_decay - 2.0) < 0.5
    assert abs(with_decay - 2.0) < 0.5
