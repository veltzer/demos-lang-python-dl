#!/usr/bin/env python

"""Tests for src/07_pytorch solutions."""

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


def test_tensor_dataset() -> None:
    mod = _load("src/07_pytorch/00_tensor_dataset.py")
    ds = mod.main()
    assert len(ds) == 100
    x, y = ds[0]
    assert tuple(x.shape) == (4,)
    assert y.dim() == 0


def test_dataloader() -> None:
    mod = _load("src/07_pytorch/01_dataloader.py")
    loader, batches = mod.main()
    assert loader.batch_size == 16
    assert len(batches) == 3
    for xb, yb in batches:
        assert tuple(xb.shape) == (16, 4)
        assert tuple(yb.shape) == (16,)


def test_custom_dataset() -> None:
    mod = _load("src/07_pytorch/02_custom_dataset.py")
    ds, batches = mod.main()
    assert len(ds) == 10
    assert len(batches) == 5
    # square relationship i -> i*i
    for idx_batch, sq_batch in batches:
        assert torch.equal(idx_batch * idx_batch, sq_batch)


def test_device_placement() -> None:
    mod = _load("src/07_pytorch/03_device_placement.py")
    device, y = mod.main()
    assert device in ("cuda", "cpu")
    assert tuple(y.shape) == (3, 2)


def test_save_load_state() -> None:
    mod = _load("src/07_pytorch/04_save_load_state.py")
    equal = mod.main()
    assert equal is True


def test_train_loop() -> None:
    mod = _load("src/07_pytorch/05_train_loop.py")
    _model, final_loss, final_acc = mod.main()
    # the synthetic argmax task is learnable; loss should drop and accuracy should be decent
    assert final_loss < 1.0
    assert final_acc > 0.6


def test_train_test_split() -> None:
    mod = _load("src/07_pytorch/06_train_test_split.py")
    train, test = mod.main()
    assert len(train) == 800
    assert len(test) == 200
