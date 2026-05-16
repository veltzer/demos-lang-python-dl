#!/usr/bin/env python

"""Tests for src/04_cnn solutions."""

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


def test_conv2d_shape() -> None:
    mod = _load("src/02_cnns/00_conv2d_shape.py")
    shape = mod.main()
    # padding=1 with k=3 preserves spatial dims; 8 out channels
    assert tuple(shape) == (2, 8, 32, 32)


def test_stride_padding() -> None:
    mod = _load("src/02_cnns/01_stride_padding.py")
    results = mod.main()
    assert len(results) == 4
    # Conv2d on 7x7 with k=3:
    # stride=1, pad=0 -> 5; pad=1 -> 7
    # stride=2, pad=0 -> 3; pad=1 -> 4
    expected = {
        (1, 0): (1, 1, 5, 5),
        (1, 1): (1, 1, 7, 7),
        (2, 0): (1, 1, 3, 3),
        (2, 1): (1, 1, 4, 4),
    }
    for stride, padding, shape in results:
        assert tuple(shape) == expected[(stride, padding)]


def test_pooling() -> None:
    mod = _load("src/02_cnns/02_pooling.py")
    x, maxp, avgp = mod.main()
    assert tuple(x.shape) == (1, 1, 4, 4)
    assert tuple(maxp.shape) == (1, 1, 2, 2)
    assert tuple(avgp.shape) == (1, 1, 2, 2)
    # MaxPool over 4x4 input (arange) with 2x2 kernel
    # top-left window: [0,1,4,5] -> max 5, avg 2.5
    assert maxp[0, 0, 0, 0].item() == 5.0
    assert avgp[0, 0, 0, 0].item() == 2.5


def test_edge_detection() -> None:
    mod = _load("src/02_cnns/03_edge_detection.py")
    img, out = mod.main()
    assert tuple(img.shape) == (5, 5)
    assert tuple(out.shape) == (5, 5)
    # The Sobel-X should produce strong response at the edge column (col 2->3 transition)
    # The vertical edge is between col 2 and col 3
    assert out[2, 2].item() > 0.0


def test_mnist_cnn() -> None:
    mod = _load("src/02_cnns/04_mnist_cnn.py")
    _model, y = mod.main()
    assert tuple(y.shape) == (4, 10)


def test_batchnorm() -> None:
    mod = _load("src/02_cnns/05_batchnorm.py")
    mean, std = mod.main()
    # BN normalizes to ~zero mean, unit std per channel
    assert torch.allclose(mean, torch.zeros(4), atol=1e-5)
    assert torch.allclose(std, torch.ones(4), atol=1e-4)


def test_global_avg_pool() -> None:
    mod = _load("src/02_cnns/06_global_avg_pool.py")
    y = mod.main()
    assert tuple(y.shape) == (2, 16)


def test_transpose_conv() -> None:
    mod = _load("src/02_cnns/07_transpose_conv.py")
    y = mod.main()
    # kernel=2 stride=2 doubles spatial dims, 4 out channels
    assert tuple(y.shape) == (1, 4, 8, 8)
