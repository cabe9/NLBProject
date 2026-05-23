from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nlb_project.models.ndt_lite import _poisson_loss, _require_torch
from nlb_project.models.stndt_axial import (
    _axial_spatiotemporal_transformer_cls,
    fit_predict_stndt_axial,
)

pytest.importorskip("torch")


def test_stndt_axial_shapes_and_positive_rates() -> None:
    rng = np.random.default_rng(401)
    train_hi = rng.poisson(0.5, (5, 8, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 8, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 8, 6)).astype(np.float32)

    out = fit_predict_stndt_axial(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        spatial_n_heads=2,
        n_spatial_latents=3,
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
        contrast_loss_weight=0.0,
        ensemble_size=1,
        device="cpu",
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (2, 8, 3)
    assert np.all(out["eval_rates_heldout"] > 0.0)


def test_stndt_axial_poisson_loss_is_finite_on_synthetic_batch() -> None:
    torch, nn, functional = _require_torch()
    torch.manual_seed(403)
    model_cls = _axial_spatiotemporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=4,
        n_heldout=2,
        d_model=8,
        max_t_len=6,
        n_layers=1,
        n_heads=2,
        spatial_n_heads=2,
        n_spatial_latents=3,
        dropout=0.0,
        min_rate=1e-6,
    )
    model = model_cls()
    x = torch.randn(3, 6, 4)
    mask = torch.rand_like(x) < 0.25
    target_hi = torch.poisson(torch.full((3, 6, 4), 0.5))
    target_ho = torch.poisson(torch.full((3, 6, 2), 0.4))

    pred_hi, pred_ho, _features = model(x.masked_fill(mask, 0.0), mask=mask)
    loss = _poisson_loss(functional, pred_ho, target_ho)
    loss = loss + 0.1 * _poisson_loss(functional, pred_hi[mask], target_hi[mask])
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert bool(torch.isfinite(loss))
    assert grads
    assert all(bool(torch.isfinite(grad).all()) for grad in grads)


def test_stndt_axial_invalid_spatial_heads_raises() -> None:
    rng = np.random.default_rng(405)
    train_hi = rng.poisson(0.5, (3, 6, 4)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 6, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 6, 4)).astype(np.float32)
    common: dict[str, Any] = {
        "d_model": 10,
        "n_layers": 1,
        "n_heads": 2,
        "n_spatial_latents": 3,
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
    with pytest.raises(ValueError, match="spatial_n_heads"):
        fit_predict_stndt_axial(
            train_hi,
            train_ho,
            eval_hi,
            spatial_n_heads=4,
            **common,
        )


def test_stndt_axial_invalid_contrast_weight_raises() -> None:
    rng = np.random.default_rng(407)
    train_hi = rng.poisson(0.5, (3, 6, 4)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 6, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 6, 4)).astype(np.float32)

    with pytest.raises(ValueError, match="contrast_loss_weight"):
        fit_predict_stndt_axial(
            train_hi,
            train_ho,
            eval_hi,
            d_model=8,
            n_layers=1,
            n_heads=2,
            spatial_n_heads=2,
            n_spatial_latents=3,
            dropout=0.0,
            learning_rate=0.003,
            weight_decay=0.0,
            batch_size=2,
            max_epochs=1,
            patience=1,
            mask_prob=0.0,
            heldin_loss_weight=0.0,
            validation_fraction=0.0,
            input_transform="sqrt_zscore",
            seed=0,
            contrast_loss_weight=-0.1,
            device="cpu",
        )
