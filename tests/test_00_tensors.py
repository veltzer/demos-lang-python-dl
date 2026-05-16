"""Tests for src/00_tensors/."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_create_tensor() -> None:
    mod = load("src/00_tensors/00_create_tensor.py")
    t = mod.main()
    assert t.tolist() == [1, 2, 3, 4, 5]
    assert t.dtype == torch.int64
    assert tuple(t.shape) == (5,)


def test_tensor_arithmetic() -> None:
    mod = load("src/00_tensors/01_tensor_arithmetic.py")
    s, p, d = mod.main()
    assert s.tolist() == [11, 22, 33]
    assert p.tolist() == [10, 40, 90]
    assert d.item() == 140


def test_reshape_and_view() -> None:
    mod = load("src/00_tensors/02_reshape_and_view.py")
    t, m, c = mod.main()
    assert tuple(t.shape) == (12,)
    assert tuple(m.shape) == (3, 4)
    assert tuple(c.shape) == (2, 2, 3)


def test_indexing_and_slicing() -> None:
    mod = load("src/00_tensors/03_indexing_and_slicing.py")
    row, col, sub = mod.main()
    assert row.tolist() == [5, 6, 7, 8, 9]
    assert col.tolist() == [4, 9, 14, 19, 24]
    assert sub.tolist() == [[6, 7, 8], [11, 12, 13], [16, 17, 18]]


def test_dtype_and_device() -> None:
    mod = load("src/00_tensors/04_dtype_and_device.py")
    t, t64, ti = mod.main()
    assert t.dtype == torch.float32
    assert t64.dtype == torch.float64
    assert ti.dtype == torch.int32


def test_random_tensors() -> None:
    mod = load("src/00_tensors/05_random_tensors.py")
    u, n, i = mod.main()
    assert tuple(u.shape) == (2, 3)
    assert tuple(n.shape) == (2, 3)
    assert tuple(i.shape) == (5,)
    assert (u >= 0).all() and (u < 1).all()
    assert (i >= 0).all() and (i < 10).all()


def test_broadcasting() -> None:
    mod = load("src/00_tensors/06_broadcasting.py")
    out = mod.main()
    assert out.tolist() == [
        [0, 11, 22, 33],
        [4, 15, 26, 37],
        [8, 19, 30, 41],
    ]


def test_matmul() -> None:
    mod = load("src/00_tensors/07_matmul.py")
    c1, c2, c3 = mod.main()
    assert torch.allclose(c1, c2)
    assert torch.allclose(c1, c3)
    assert c1.tolist() == [[4.0, 5.0], [10.0, 11.0]]


def test_reductions() -> None:
    mod = load("src/00_tensors/08_reductions.py")
    s, cs, rm, am = mod.main()
    assert tuple(cs.shape) == (4,)
    assert tuple(rm.shape) == (3,)
    assert s.item() > 0
    assert 0 <= am.item() < 12


def test_numpy_bridge() -> None:
    mod = load("src/00_tensors/09_numpy_bridge.py")
    arr, t = mod.main()
    assert isinstance(arr, np.ndarray)
    assert isinstance(t, torch.Tensor)
    assert arr.tolist() == [2.0, 4.0, 6.0]
    assert t.tolist() == [2.0, 4.0, 6.0]
