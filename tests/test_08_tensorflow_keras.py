"""Tests for src/08_tensorflow_keras/* solutions.

tensorflow / keras may not be installed; every test importorskip's them.
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


def test_00_tf_tensor():
    pytest.importorskip("tensorflow")
    mod = load("src/08_tensorflow_keras/00_tf_tensor.py")
    t = mod.main()
    assert tuple(t.shape) == (5,)


def test_01_keras_sequential():
    pytest.importorskip("keras")
    mod = load("src/08_tensorflow_keras/01_keras_sequential.py")
    model = mod.main()
    assert len(model.layers) == 3


def test_02_compile_fit():
    pytest.importorskip("keras")
    mod = load("src/08_tensorflow_keras/02_compile_fit.py")
    _model, history = mod.main()
    assert "loss" in history.history
    assert len(history.history["loss"]) == 5


def test_03_functional_api():
    pytest.importorskip("keras")
    mod = load("src/08_tensorflow_keras/03_functional_api.py")
    model = mod.main()
    assert len(model.inputs) == 2


def test_04_cnn_keras():
    pytest.importorskip("keras")
    mod = load("src/08_tensorflow_keras/04_cnn_keras.py")
    model = mod.main()
    assert model.output_shape == (None, 10)


def test_05_save_load():
    pytest.importorskip("keras")
    mod = load("src/08_tensorflow_keras/05_save_load.py")
    same = mod.main()
    assert same is True


def test_06_gradient_tape():
    pytest.importorskip("tensorflow")
    mod = load("src/08_tensorflow_keras/06_gradient_tape.py")
    grad = mod.main()
    # d/dx x^3 at x=4 is 48
    assert float(grad.numpy()) == pytest.approx(48.0)
