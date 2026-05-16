"""Tests for src/11_training_techniques/* solutions."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_00_dropout():
    mod = load("src/11_training_techniques/00_dropout.py")
    t1, t2, e1, e2 = mod.main()
    # Train mode is stochastic -> outputs differ; eval mode is deterministic.
    assert not torch.allclose(t1, t2)
    assert torch.allclose(e1, e2)


def test_01_layernorm():
    mod = load("src/11_training_techniques/01_layernorm.py")
    means, stds = mod.main()
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)


def test_02_grad_clipping():
    mod = load("src/11_training_techniques/02_grad_clipping.py")
    before, after = mod.main()
    assert before > 1e5
    assert after == pytest.approx(1.0, abs=1e-5)


def test_03_early_stopping():
    mod = load("src/11_training_techniques/03_early_stopping.py")
    stopped_at, best = mod.main()
    assert stopped_at > 0
    assert best < float("inf")


def test_04_weight_init():
    mod = load("src/11_training_techniques/04_weight_init.py")
    results = mod.main()
    assert set(results) == {"normal(0, 0.01)", "xavier_normal", "kaiming_normal"}
    # normal(0, 0.01) should produce std ~0.01
    assert abs(results["normal(0, 0.01)"] - 0.01) < 0.005
    # xavier with fan_in=fan_out=100 → std = sqrt(2/200) ≈ 0.1
    assert 0.05 < results["xavier_normal"] < 0.2
    # kaiming relu with fan_in=100 → std ≈ sqrt(2/100) ≈ 0.14
    assert 0.05 < results["kaiming_normal"] < 0.25


def test_05_overfit_small_batch():
    mod = load("src/11_training_techniques/05_overfit_small_batch.py")
    final_loss = mod.main()
    # Should drive loss near zero on 8 samples
    assert final_loss < 0.05


def test_06_label_smoothing():
    mod = load("src/11_training_techniques/06_label_smoothing.py")
    plain, smooth = mod.main()
    # Label smoothing raises the loss on a confident-correct prediction
    assert smooth.item() > plain.item()
