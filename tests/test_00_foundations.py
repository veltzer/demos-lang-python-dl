"""Tests for src/00_foundations/* (autograd, MLP basics, optimizers, training techniques)."""

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn


def _load(path: str) -> ModuleType:
    p = Path(__file__).parent.parent / path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load = _load




def test_requires_grad() -> None:
    mod = _load("src/00_foundations/00_requires_grad.py")
    grad = mod.main()
    # d/dx (x^2) at x=3 is 6
    assert grad.item() == 6.0


def test_vector_grad() -> None:
    mod = _load("src/00_foundations/01_vector_grad.py")
    grad = mod.main()
    # d/dx_i sum(x^2) = 2*x_i
    assert torch.allclose(grad, torch.tensor([2.0, 4.0, 6.0]))


def test_chain_rule() -> None:
    mod = _load("src/00_foundations/02_chain_rule.py")
    grad = mod.main()
    # y = 3x, z = y^2 + y = 9x^2 + 3x, dz/dx = 18x + 3; at x=2 -> 39
    assert grad.item() == 39.0


def test_no_grad() -> None:
    mod = _load("src/00_foundations/03_no_grad.py")
    y_req, z_req = mod.main()
    assert y_req is False
    assert z_req is False


def test_zero_grad() -> None:
    mod = _load("src/00_foundations/04_zero_grad.py")
    g1, g2, g3 = mod.main()
    # x=2, d/dx x^2 = 2x = 4. Without zero_grad, second backward accumulates to 8.
    assert g1.item() == 4.0
    assert g2.item() == 8.0
    assert g3.item() == 4.0


def test_jacobian() -> None:
    mod = _load("src/00_foundations/05_jacobian.py")
    j = mod.main()
    # x=[2,3]; f = [x0^2, x0*x1, x1^3]
    # Jacobian: [[2x0, 0], [x1, x0], [0, 3x1^2]] = [[4,0],[3,2],[0,27]]
    expected = torch.tensor([[4.0, 0.0], [3.0, 2.0], [0.0, 27.0]])
    assert torch.allclose(j, expected)


def test_manual_gradient_descent() -> None:
    mod = _load("src/00_foundations/06_manual_gradient_descent.py")
    val = mod.main()
    # Should converge near 5.0
    assert abs(val - 5.0) < 1e-3


def test_grad_of_grad() -> None:
    mod = _load("src/00_foundations/07_grad_of_grad.py")
    g1, g2 = mod.main()
    # y = x^3; y' = 3x^2 = 12 at x=2; y'' = 6x = 12 at x=2
    assert g1 == 12.0
    assert g2 == 12.0



def test_linear_layer() -> None:
    mod = _load("src/00_foundations/08_linear_layer.py")
    layer, y = mod.main()
    assert tuple(y.shape) == (4, 2)
    assert tuple(layer.weight.shape) == (2, 3)
    assert tuple(layer.bias.shape) == (2,)


def test_activations() -> None:
    mod = _load("src/00_foundations/09_activations.py")
    relu, sigmoid, tanh, softmax = mod.main()
    assert torch.allclose(relu, torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    # sigmoid(0) = 0.5
    assert abs(sigmoid[2].item() - 0.5) < 1e-6
    # tanh(0) = 0
    assert abs(tanh[2].item()) < 1e-6
    # softmax should sum to 1
    assert abs(softmax.sum().item() - 1.0) < 1e-6


def test_sequential() -> None:
    mod = _load("src/00_foundations/10_sequential.py")
    model, y = mod.main()
    assert tuple(y.shape) == (8, 2)
    assert isinstance(model, nn.Sequential)


def test_custom_module() -> None:
    mod = _load("src/00_foundations/11_custom_module.py")
    model, y = mod.main()
    assert tuple(y.shape) == (5, 3)
    assert hasattr(model, "linear1")
    assert hasattr(model, "linear2")


def test_count_parameters() -> None:
    mod = _load("src/00_foundations/12_count_parameters.py")
    total = mod.main()
    # Linear(20,100): 20*100 + 100 = 2100
    # Linear(100,50): 100*50 + 50 = 5050
    # Linear(50,10): 50*10 + 10 = 510
    # total = 7660
    assert total == 7660


def test_loss_functions() -> None:
    mod = _load("src/00_foundations/13_loss_functions.py")
    mse_val, bce_val, ce_val = mod.main()
    # MSE of difference 0.5 across all entries -> 0.25
    assert abs(mse_val.item() - 0.25) < 1e-6
    assert bce_val.item() > 0.0
    assert ce_val.item() > 0.0


def test_xor_problem() -> None:
    mod = _load("src/00_foundations/14_xor_problem.py")
    preds = mod.main()
    expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.allclose(preds, expected)


def test_train_eval_mode() -> None:
    mod = _load("src/00_foundations/15_train_eval_mode.py")
    t1, t2, e1, e2 = mod.main()
    # In train mode, dropout makes outputs differ between calls
    assert not torch.allclose(t1, t2)
    # In eval mode, dropout is a no-op, so two passes are identical
    assert torch.allclose(e1, e2)



def test_sgd_minimize() -> None:
    mod = _load("src/00_foundations/16_sgd_minimize.py")
    val = mod.main()
    # Should converge close to 7
    assert abs(val - 7.0) < 1e-3


def test_adam_minimize() -> None:
    mod = _load("src/00_foundations/17_adam_minimize.py")
    p = mod.main()
    # Should converge near (3, -4)
    assert abs(p[0].item() - 3.0) < 0.05
    assert abs(p[1].item() + 4.0) < 0.05


def test_momentum() -> None:
    mod = _load("src/00_foundations/18_momentum.py")
    plain, momentum = mod.main()
    assert not math.isnan(plain)
    assert not math.isnan(momentum)


def test_lr_scheduler() -> None:
    mod = _load("src/00_foundations/19_lr_scheduler.py")
    lrs = mod.main()
    assert len(lrs) == 20
    # lr starts 0.1, halves every 5 steps
    # After 5 steps -> 0.05, after 10 -> 0.025, after 15 -> 0.0125, after 20 -> 0.00625
    assert abs(lrs[0] - 0.1) < 1e-9
    assert abs(lrs[4] - 0.05) < 1e-9
    assert abs(lrs[9] - 0.025) < 1e-9
    assert abs(lrs[14] - 0.0125) < 1e-9
    assert abs(lrs[19] - 0.00625) < 1e-9


def test_linear_regression() -> None:
    mod = _load("src/00_foundations/20_linear_regression.py")
    w, b = mod.main()
    # True w=2, b=1; with noise we allow some tolerance
    assert abs(w - 2.0) < 0.1
    assert abs(b - 1.0) < 0.3


def test_minibatch() -> None:
    mod = _load("src/00_foundations/21_minibatch.py")
    w, b = mod.main()
    # True w=2, b=1; minibatch should still get reasonably close
    assert abs(w - 2.0) < 0.2
    assert abs(b - 1.0) < 0.5


def test_weight_decay() -> None:
    mod = _load("src/00_foundations/22_weight_decay.py")
    no_decay, with_decay = mod.main()
    # Both should still be in a reasonable neighbourhood of the true slope (2.0).
    # weight_decay biases toward zero but with this data/seed both end up near 2.
    assert abs(no_decay - 2.0) < 0.5
    assert abs(with_decay - 2.0) < 0.5



def test_00_dropout():
    mod = load("src/00_foundations/23_dropout.py")
    t1, t2, e1, e2 = mod.main()
    # Train mode is stochastic -> outputs differ; eval mode is deterministic.
    assert not torch.allclose(t1, t2)
    assert torch.allclose(e1, e2)


def test_01_layernorm():
    mod = load("src/00_foundations/24_layernorm.py")
    means, stds = mod.main()
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)


def test_02_grad_clipping():
    mod = load("src/00_foundations/25_grad_clipping.py")
    before, after = mod.main()
    assert before > 1e5
    assert after == pytest.approx(1.0, abs=1e-5)


def test_03_early_stopping():
    mod = load("src/00_foundations/26_early_stopping.py")
    stopped_at, best = mod.main()
    assert stopped_at > 0
    assert best < float("inf")


def test_04_weight_init():
    mod = load("src/00_foundations/27_weight_init.py")
    results = mod.main()
    assert set(results) == {"normal(0, 0.01)", "xavier_normal", "kaiming_normal"}
    # normal(0, 0.01) should produce std ~0.01
    assert abs(results["normal(0, 0.01)"] - 0.01) < 0.005
    # xavier with fan_in=fan_out=100 → std = sqrt(2/200) ≈ 0.1
    assert 0.05 < results["xavier_normal"] < 0.2
    # kaiming relu with fan_in=100 → std ≈ sqrt(2/100) ≈ 0.14
    assert 0.05 < results["kaiming_normal"] < 0.25


def test_05_overfit_small_batch():
    mod = load("src/00_foundations/28_overfit_small_batch.py")
    final_loss = mod.main()
    # Should drive loss near zero on 8 samples
    assert final_loss < 0.05


def test_06_label_smoothing():
    mod = load("src/00_foundations/29_label_smoothing.py")
    plain, smooth = mod.main()
    # Label smoothing raises the loss on a confident-correct prediction
    assert smooth.item() > plain.item()
