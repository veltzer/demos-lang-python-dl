"""Tests for src/01_frameworks/* (PyTorch tensors + data + tf/keras + jax/flax)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest
import torch


def _load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load = _load




def test_create_tensor() -> None:
    mod = load("src/01_frameworks/00_create_tensor.py")
    t = mod.main()
    assert t.tolist() == [1, 2, 3, 4, 5]
    assert t.dtype == torch.int64
    assert tuple(t.shape) == (5,)


def test_tensor_arithmetic() -> None:
    mod = load("src/01_frameworks/01_tensor_arithmetic.py")
    s, p, d = mod.main()
    assert s.tolist() == [11, 22, 33]
    assert p.tolist() == [10, 40, 90]
    assert d.item() == 140


def test_reshape_and_view() -> None:
    mod = load("src/01_frameworks/02_reshape_and_view.py")
    t, m, c = mod.main()
    assert tuple(t.shape) == (12,)
    assert tuple(m.shape) == (3, 4)
    assert tuple(c.shape) == (2, 2, 3)


def test_indexing_and_slicing() -> None:
    mod = load("src/01_frameworks/03_indexing_and_slicing.py")
    row, col, sub = mod.main()
    assert row.tolist() == [5, 6, 7, 8, 9]
    assert col.tolist() == [4, 9, 14, 19, 24]
    assert sub.tolist() == [[6, 7, 8], [11, 12, 13], [16, 17, 18]]


def test_dtype_and_device() -> None:
    mod = load("src/01_frameworks/04_dtype_and_device.py")
    t, t64, ti = mod.main()
    assert t.dtype == torch.float32
    assert t64.dtype == torch.float64
    assert ti.dtype == torch.int32


def test_random_tensors() -> None:
    mod = load("src/01_frameworks/05_random_tensors.py")
    u, n, i = mod.main()
    assert tuple(u.shape) == (2, 3)
    assert tuple(n.shape) == (2, 3)
    assert tuple(i.shape) == (5,)
    assert (u >= 0).all() and (u < 1).all()
    assert (i >= 0).all() and (i < 10).all()


def test_broadcasting() -> None:
    mod = load("src/01_frameworks/06_broadcasting.py")
    out = mod.main()
    assert out.tolist() == [
        [0, 11, 22, 33],
        [4, 15, 26, 37],
        [8, 19, 30, 41],
    ]


def test_matmul() -> None:
    mod = load("src/01_frameworks/07_matmul.py")
    c1, c2, c3 = mod.main()
    assert torch.allclose(c1, c2)
    assert torch.allclose(c1, c3)
    assert c1.tolist() == [[4.0, 5.0], [10.0, 11.0]]


def test_reductions() -> None:
    mod = load("src/01_frameworks/08_reductions.py")
    s, cs, rm, am = mod.main()
    assert tuple(cs.shape) == (4,)
    assert tuple(rm.shape) == (3,)
    assert s.item() > 0
    assert 0 <= am.item() < 12


def test_numpy_bridge() -> None:
    mod = load("src/01_frameworks/09_numpy_bridge.py")
    arr, t = mod.main()
    assert isinstance(arr, np.ndarray)
    assert isinstance(t, torch.Tensor)
    assert arr.tolist() == [2.0, 4.0, 6.0]
    assert t.tolist() == [2.0, 4.0, 6.0]



def test_tensor_dataset() -> None:
    mod = _load("src/01_frameworks/10_tensor_dataset.py")
    ds = mod.main()
    assert len(ds) == 100
    x, y = ds[0]
    assert tuple(x.shape) == (4,)
    assert y.dim() == 0


def test_dataloader() -> None:
    mod = _load("src/01_frameworks/11_dataloader.py")
    loader, batches = mod.main()
    assert loader.batch_size == 16
    assert len(batches) == 3
    for xb, yb in batches:
        assert tuple(xb.shape) == (16, 4)
        assert tuple(yb.shape) == (16,)


def test_custom_dataset() -> None:
    mod = _load("src/01_frameworks/12_custom_dataset.py")
    ds, batches = mod.main()
    assert len(ds) == 10
    assert len(batches) == 5
    # square relationship i -> i*i
    for idx_batch, sq_batch in batches:
        assert torch.equal(idx_batch * idx_batch, sq_batch)


def test_device_placement() -> None:
    mod = _load("src/01_frameworks/13_device_placement.py")
    device, y = mod.main()
    assert device in ("cuda", "cpu")
    assert tuple(y.shape) == (3, 2)


def test_save_load_state() -> None:
    mod = _load("src/01_frameworks/14_save_load_state.py")
    equal = mod.main()
    assert equal is True


def test_train_loop() -> None:
    mod = _load("src/01_frameworks/15_train_loop.py")
    _model, final_loss, final_acc = mod.main()
    # the synthetic argmax task is learnable; loss should drop and accuracy should be decent
    assert final_loss < 1.0
    assert final_acc > 0.6


def test_train_test_split() -> None:
    mod = _load("src/01_frameworks/16_train_test_split.py")
    train, test = mod.main()
    assert len(train) == 800
    assert len(test) == 200



def test_00_tf_tensor():
    pytest.importorskip("tensorflow")
    mod = load("src/01_frameworks/17_tf_tensor.py")
    t = mod.main()
    assert tuple(t.shape) == (5,)


def test_01_keras_sequential():
    pytest.importorskip("keras")
    mod = load("src/01_frameworks/18_keras_sequential.py")
    model = mod.main()
    assert len(model.layers) == 3


def test_02_compile_fit():
    pytest.importorskip("keras")
    mod = load("src/01_frameworks/19_compile_fit.py")
    _model, history = mod.main()
    assert "loss" in history.history
    assert len(history.history["loss"]) == 5


def test_03_functional_api():
    pytest.importorskip("keras")
    mod = load("src/01_frameworks/20_functional_api.py")
    model = mod.main()
    assert len(model.inputs) == 2


def test_04_cnn_keras():
    pytest.importorskip("keras")
    mod = load("src/01_frameworks/21_cnn_keras.py")
    model = mod.main()
    assert model.output_shape == (None, 10)


def test_05_save_load():
    pytest.importorskip("keras")
    mod = load("src/01_frameworks/22_save_load_keras.py")
    same = mod.main()
    assert same is True


def test_06_gradient_tape():
    pytest.importorskip("tensorflow")
    mod = load("src/01_frameworks/23_gradient_tape.py")
    grad = mod.main()
    # d/dx x^3 at x=4 is 48
    assert float(grad.numpy()) == pytest.approx(48.0)



def test_00_jnp_basics():
    mod = load("src/01_frameworks/24_jnp_basics.py")
    doubled, total = mod.main()
    assert jnp.allclose(doubled, jnp.array([2.0, 4.0, 6.0]))
    assert float(total) == pytest.approx(6.0)


def test_01_grad():
    mod = load("src/01_frameworks/25_grad.py")
    g = mod.main()
    # d/dx (x^3 + 2x) at x=3 is 3*9 + 2 = 29
    assert float(g) == pytest.approx(29.0)


def test_02_jit():
    mod = load("src/01_frameworks/26_jit.py")
    result = mod.main()
    # sum_{k=0}^{999} sin(k)^2 ≈ 500
    assert float(result) == pytest.approx(500.0, abs=5.0)


def test_03_vmap():
    mod = load("src/01_frameworks/27_vmap.py")
    out = mod.main()
    assert out.shape == (5,)


def test_04_prng():
    mod = load("src/01_frameworks/28_prng.py")
    s1, s2, s3 = mod.main()
    assert s1.shape == (2, 2)
    assert s2.shape == (2, 2)
    assert s3.shape == (2, 2)
    # Distinct keys produce distinct samples
    assert not jnp.allclose(s1, s2)
    assert not jnp.allclose(s2, s3)


def test_05_flax_linen():
    mod = load("src/01_frameworks/29_flax_linen.py")
    _model, _params, out = mod.main()
    assert out.shape == (4, 3)


def test_06_train_step():
    mod = load("src/01_frameworks/30_train_step.py")
    _params, final_loss = mod.main()
    # Should fit a perfect linear y = 2x + 1 well below 1.0
    assert float(final_loss) < 1.0
