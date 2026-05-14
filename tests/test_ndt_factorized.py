from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nlb_project.models.ndt_factorized import (
    _factorized_transformer_cls,
    fit_predict_ndt_factorized,
)
from nlb_project.models.ndt_lite import _poisson_loss, _require_torch

pytest.importorskip("torch")


def test_ndt_factorized_shapes_and_positive_rates() -> None:
    rng = np.random.default_rng(191)
    train_hi = rng.poisson(0.5, (5, 12, 7)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 12, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 12, 7)).astype(np.float32)

    out = fit_predict_ndt_factorized(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_latents=3,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=2,
        patience=2,
        mask_prob=0.25,
        heldin_loss_weight=0.1,
        validation_fraction=0.2,
        input_transform="sqrt_zscore",
        seed=0,
        ensemble_size=1,
        device="cpu",
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (2, 12, 3)
    assert np.all(out["eval_rates_heldout"] > 0.0)


def test_ndt_factorized_poisson_loss_is_finite_on_synthetic_batch() -> None:
    torch, nn, functional = _require_torch()
    torch.manual_seed(195)
    model_cls = _factorized_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=4,
        n_heldout=2,
        d_model=8,
        max_t_len=6,
        n_layers=1,
        n_heads=2,
        n_latents=3,
        dropout=0.0,
        min_rate=1e-6,
    )
    model = model_cls()
    x = torch.randn(3, 6, 4)
    target_hi = torch.poisson(torch.full((3, 6, 4), 0.5))
    target_ho = torch.poisson(torch.full((3, 6, 2), 0.4))

    pred_hi, pred_ho = model(x)
    loss = _poisson_loss(functional, pred_ho, target_ho)
    loss = loss + 0.1 * _poisson_loss(functional, pred_hi, target_hi)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert bool(torch.isfinite(loss))
    assert grads
    assert all(bool(torch.isfinite(grad).all()) for grad in grads)


def test_ndt_factorized_invalid_n_latents_raises() -> None:
    rng = np.random.default_rng(197)
    train_hi = rng.poisson(0.5, (3, 6, 4)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 6, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 6, 4)).astype(np.float32)
    common: dict[str, Any] = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_epochs": 1,
        "patience": 1,
        "mask_prob": 0.0,
        "heldin_loss_weight": 0.0,
        "validation_fraction": 0.0,
        "input_transform": "sqrt_zscore",
        "seed": 0,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="n_latents"):
        fit_predict_ndt_factorized(
            train_hi,
            train_ho,
            eval_hi,
            n_latents=0,
            **common,
        )
