#!/usr/bin/env python

"""Tests for src/02_nn_basics solutions."""

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


def _load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_linear_layer() -> None:
    mod = _load("src/02_nn_basics/00_linear_layer.py")
    layer, y = mod.main()
    assert tuple(y.shape) == (4, 2)
    assert tuple(layer.weight.shape) == (2, 3)
    assert tuple(layer.bias.shape) == (2,)


def test_activations() -> None:
    mod = _load("src/02_nn_basics/01_activations.py")
    relu, sigmoid, tanh, softmax = mod.main()
    assert torch.allclose(relu, torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    # sigmoid(0) = 0.5
    assert abs(sigmoid[2].item() - 0.5) < 1e-6
    # tanh(0) = 0
    assert abs(tanh[2].item()) < 1e-6
    # softmax should sum to 1
    assert abs(softmax.sum().item() - 1.0) < 1e-6


def test_sequential() -> None:
    mod = _load("src/02_nn_basics/02_sequential.py")
    model, y = mod.main()
    assert tuple(y.shape) == (8, 2)
    assert isinstance(model, nn.Sequential)


def test_custom_module() -> None:
    mod = _load("src/02_nn_basics/03_custom_module.py")
    model, y = mod.main()
    assert tuple(y.shape) == (5, 3)
    assert hasattr(model, "linear1")
    assert hasattr(model, "linear2")


def test_count_parameters() -> None:
    mod = _load("src/02_nn_basics/04_count_parameters.py")
    total = mod.main()
    # Linear(20,100): 20*100 + 100 = 2100
    # Linear(100,50): 100*50 + 50 = 5050
    # Linear(50,10): 50*10 + 10 = 510
    # total = 7660
    assert total == 7660


def test_loss_functions() -> None:
    mod = _load("src/02_nn_basics/05_loss_functions.py")
    mse_val, bce_val, ce_val = mod.main()
    # MSE of difference 0.5 across all entries -> 0.25
    assert abs(mse_val.item() - 0.25) < 1e-6
    assert bce_val.item() > 0.0
    assert ce_val.item() > 0.0


def test_xor_problem() -> None:
    mod = _load("src/02_nn_basics/06_xor_problem.py")
    preds = mod.main()
    expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.allclose(preds, expected)


def test_train_eval_mode() -> None:
    mod = _load("src/02_nn_basics/07_train_eval_mode.py")
    t1, t2, e1, e2 = mod.main()
    # In train mode, dropout makes outputs differ between calls
    assert not torch.allclose(t1, t2)
    # In eval mode, dropout is a no-op, so two passes are identical
    assert torch.allclose(e1, e2)
