"""Tests for src/09_jax_flax/* solutions."""

import importlib.util
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import pytest


def load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_00_jnp_basics():
    mod = load("src/09_jax_flax/00_jnp_basics.py")
    doubled, total = mod.main()
    assert jnp.allclose(doubled, jnp.array([2.0, 4.0, 6.0]))
    assert float(total) == pytest.approx(6.0)


def test_01_grad():
    mod = load("src/09_jax_flax/01_grad.py")
    g = mod.main()
    # d/dx (x^3 + 2x) at x=3 is 3*9 + 2 = 29
    assert float(g) == pytest.approx(29.0)


def test_02_jit():
    mod = load("src/09_jax_flax/02_jit.py")
    result = mod.main()
    # sum_{k=0}^{999} sin(k)^2 ≈ 500
    assert float(result) == pytest.approx(500.0, abs=5.0)


def test_03_vmap():
    mod = load("src/09_jax_flax/03_vmap.py")
    out = mod.main()
    assert out.shape == (5,)


def test_04_prng():
    mod = load("src/09_jax_flax/04_prng.py")
    s1, s2, s3 = mod.main()
    assert s1.shape == (2, 2)
    assert s2.shape == (2, 2)
    assert s3.shape == (2, 2)
    # Distinct keys produce distinct samples
    assert not jnp.allclose(s1, s2)
    assert not jnp.allclose(s2, s3)


def test_05_flax_linen():
    mod = load("src/09_jax_flax/05_flax_linen.py")
    _model, _params, out = mod.main()
    assert out.shape == (4, 3)


def test_06_train_step():
    mod = load("src/09_jax_flax/06_train_step.py")
    _params, final_loss = mod.main()
    # Should fit a perfect linear y = 2x + 1 well below 1.0
    assert float(final_loss) < 1.0
