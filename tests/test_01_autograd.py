#!/usr/bin/env python

"""Tests for src/01_autograd solutions."""

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


def test_requires_grad() -> None:
    mod = _load("src/01_autograd/00_requires_grad.py")
    grad = mod.main()
    # d/dx (x^2) at x=3 is 6
    assert grad.item() == 6.0


def test_vector_grad() -> None:
    mod = _load("src/01_autograd/01_vector_grad.py")
    grad = mod.main()
    # d/dx_i sum(x^2) = 2*x_i
    assert torch.allclose(grad, torch.tensor([2.0, 4.0, 6.0]))


def test_chain_rule() -> None:
    mod = _load("src/01_autograd/02_chain_rule.py")
    grad = mod.main()
    # y = 3x, z = y^2 + y = 9x^2 + 3x, dz/dx = 18x + 3; at x=2 -> 39
    assert grad.item() == 39.0


def test_no_grad() -> None:
    mod = _load("src/01_autograd/03_no_grad.py")
    y_req, z_req = mod.main()
    assert y_req is False
    assert z_req is False


def test_zero_grad() -> None:
    mod = _load("src/01_autograd/04_zero_grad.py")
    g1, g2, g3 = mod.main()
    # x=2, d/dx x^2 = 2x = 4. Without zero_grad, second backward accumulates to 8.
    assert g1.item() == 4.0
    assert g2.item() == 8.0
    assert g3.item() == 4.0


def test_jacobian() -> None:
    mod = _load("src/01_autograd/05_jacobian.py")
    j = mod.main()
    # x=[2,3]; f = [x0^2, x0*x1, x1^3]
    # Jacobian: [[2x0, 0], [x1, x0], [0, 3x1^2]] = [[4,0],[3,2],[0,27]]
    expected = torch.tensor([[4.0, 0.0], [3.0, 2.0], [0.0, 27.0]])
    assert torch.allclose(j, expected)


def test_manual_gradient_descent() -> None:
    mod = _load("src/01_autograd/06_manual_gradient_descent.py")
    val = mod.main()
    # Should converge near 5.0
    assert abs(val - 5.0) < 1e-3


def test_grad_of_grad() -> None:
    mod = _load("src/01_autograd/07_grad_of_grad.py")
    g1, g2 = mod.main()
    # y = x^3; y' = 3x^2 = 12 at x=2; y'' = 6x = 12 at x=2
    assert g1 == 12.0
    assert g2 == 12.0
