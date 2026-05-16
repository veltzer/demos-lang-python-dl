"""Tests for src/12_generative_models/* solutions.

Stochastic models use loose tolerances (see comments per test).
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType


def load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_00_autoencoder():
    mod = load("src/12_generative_models/00_autoencoder.py")
    final_loss = mod.main()
    # 200x8 random data, bottleneck-2 -> nonzero floor, but should fall well below 1.0
    assert final_loss < 1.0


def test_01_denoising_autoencoder():
    mod = load("src/12_generative_models/01_denoising_autoencoder.py")
    clean_loss = mod.main()
    assert clean_loss < 1.0


def test_02_vae():
    mod = load("src/12_generative_models/02_vae.py")
    final_loss = mod.main()
    assert final_loss < 2.0


def test_03_gan_basic():
    """Tiny 1-D GAN — very flaky; use loose tolerances per task instructions."""
    mod = load("src/12_generative_models/03_gan_basic.py")
    mean, std = mod.main()
    # Target is N(5, 1). Tolerances: mean within ~1.5 of 5.0, std within ~1.0 of 1.0.
    assert abs(mean - 5.0) < 1.5
    assert abs(std - 1.0) < 1.0


def test_04_normalizing_flow():
    """Affine flow should learn s ≈ log(2), t ≈ 3."""
    mod = load("src/12_generative_models/04_normalizing_flow.py")
    s, t = mod.main()
    assert abs(s - math.log(2.0)) < 0.5
    assert abs(t - 3.0) < 0.5


def test_05_diffusion_basic():
    mod = load("src/12_generative_models/05_diffusion_basic.py")
    final_loss = mod.main()
    # Final-step loss on a single batch is noisy but the average should be < 1.0
    assert final_loss < 1.0
