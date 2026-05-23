from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nlb_project.models.ndt_lite import (
    _poisson_loss,
    _require_torch,
    _temporal_transformer_cls,
    fit_predict_ndt_lite,
)

pytest.importorskip("torch")


def test_ndt_lite_shapes_and_positive_rates() -> None:
    rng = np.random.default_rng(91)
    train_hi = rng.poisson(0.5, (5, 12, 7)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 12, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 12, 7)).astype(np.float32)

    out = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
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


def test_ndt_lite_seed_ensemble_shapes_and_positive_rates() -> None:
    rng = np.random.default_rng(92)
    train_hi = rng.poisson(0.5, (4, 8, 5)).astype(np.float32)
    train_ho = rng.poisson(0.4, (4, 8, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 8, 5)).astype(np.float32)

    out = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=1,
        patience=1,
        mask_prob=0.25,
        heldin_loss_weight=0.1,
        validation_fraction=0.25,
        input_transform="sqrt_zscore",
        seed=0,
        ensemble_size=2,
        device="cpu",
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldin"].shape == eval_hi.shape
    assert out["eval_rates_heldout"].shape == (2, 8, 2)
    assert np.all(out["eval_rates_heldout"] > 0.0)


def test_ndt_lite_poisson_loss_is_finite_on_synthetic_batch() -> None:
    torch, nn, functional = _require_torch()
    torch.manual_seed(95)
    model_cls = _temporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=4,
        n_heldout=2,
        d_model=8,
        max_t_len=6,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        min_rate=1e-6,
        neuron_embedding_scale=0.0,
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


def test_ndt_lite_invalid_ensemble_size_raises() -> None:
    rng = np.random.default_rng(93)
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
    with pytest.raises(ValueError, match="ensemble_size"):
        fit_predict_ndt_lite(
            train_hi,
            train_ho,
            eval_hi,
            ensemble_size=0,
            **common,
        )


def test_ndt_lite_invalid_lr_schedule_raises() -> None:
    rng = np.random.default_rng(96)
    train_hi = rng.poisson(0.5, (3, 6, 4)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 6, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 6, 4)).astype(np.float32)

    with pytest.raises(ValueError, match="lr_schedule"):
        fit_predict_ndt_lite(
            train_hi,
            train_ho,
            eval_hi,
            d_model=8,
            n_layers=1,
            n_heads=2,
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
            lr_schedule="not_a_schedule",
            device="cpu",
        )


def test_ndt_lite_invalid_neuron_embedding_scale_raises() -> None:
    rng = np.random.default_rng(97)
    train_hi = rng.poisson(0.5, (3, 6, 4)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 6, 2)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 6, 4)).astype(np.float32)

    with pytest.raises(ValueError, match="neuron_embedding_scale"):
        fit_predict_ndt_lite(
            train_hi,
            train_ho,
            eval_hi,
            d_model=8,
            n_layers=1,
            n_heads=2,
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
            neuron_embedding_scale=-0.1,
            device="cpu",
        )


def test_neuron_event_embeddings_change_transformer_output() -> None:
    torch, nn, functional = _require_torch()
    torch.manual_seed(98)
    model_cls = _temporal_transformer_cls(
        nn,
        functional,
        torch,
        n_heldin=4,
        n_heldout=2,
        d_model=8,
        max_t_len=6,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        min_rate=1e-6,
        neuron_embedding_scale=0.5,
    )
    model = model_cls()
    model.eval()
    x = torch.randn(2, 6, 4)
    raw_counts = torch.zeros(2, 6, 4)
    raw_counts[:, :, 0] = 1.0

    with torch.no_grad():
        _heldin_a, heldout_a = model(x, raw_counts)
        _heldin_b, heldout_b = model(x, torch.zeros_like(raw_counts))

    assert not torch.allclose(heldout_a, heldout_b)


def test_ndt_lite_ensemble_matches_mean_of_single_seed_runs() -> None:
    """Members use seeds ``seed``, ``seed+1``, …; ensemble output averages their rates."""
    rng = np.random.default_rng(94)
    train_hi = rng.poisson(0.5, (5, 10, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 10, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 10, 6)).astype(np.float32)
    kw: dict[str, Any] = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": 3,
        "max_epochs": 2,
        "patience": 2,
        "mask_prob": 0.2,
        "heldin_loss_weight": 0.1,
        "validation_fraction": 0.2,
        "input_transform": "sqrt_zscore",
        "seed": 7,
        "device": "cpu",
    }
    out_ens = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        ensemble_size=2,
        **kw,
    )
    out_a = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        ensemble_size=1,
        seed=7,
        **{k: v for k, v in kw.items() if k != "seed"},
    )
    out_b = fit_predict_ndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        ensemble_size=1,
        seed=8,
        **{k: v for k, v in kw.items() if k != "seed"},
    )
    for key in out_ens:
        mean_manual = 0.5 * (out_a[key] + out_b[key])
        assert np.allclose(out_ens[key], mean_manual, rtol=1e-5, atol=1e-5)
        assert np.all(out_ens[key] > 0.0)
