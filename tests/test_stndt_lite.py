from __future__ import annotations

import numpy as np
import pytest


def test_sample_block_time_mask_all_neurons_share_time_bins() -> None:
    torch = pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import sample_block_time_mask

    torch.manual_seed(0)
    mask = sample_block_time_mask(
        batch_size=4,
        t_len=32,
        n_neurons=6,
        mask_prob=0.6,
        span_length=4,
        device=torch.device("cpu"),
    )

    assert mask.shape == (4, 32, 6)
    assert mask.dtype == torch.bool
    per_time = mask.all(dim=-1) == mask.any(dim=-1)
    assert bool(per_time.all())

    masked_rate = float(mask.float().mean().item())
    assert 0.45 <= masked_rate <= 0.75


def test_sample_block_time_mask_span_covers_full_sequence() -> None:
    torch = pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import sample_block_time_mask

    mask = sample_block_time_mask(
        batch_size=2,
        t_len=8,
        n_neurons=3,
        mask_prob=0.5,
        span_length=8,
        device=torch.device("cpu"),
    )

    assert bool(mask.all())


def test_fit_predict_stndt_lite_block_time_runs() -> None:
    pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import fit_predict_stndt_lite

    rng = np.random.default_rng(501)
    train_hi = rng.poisson(0.5, (5, 8, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 8, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 8, 6)).astype(np.float32)

    out = fit_predict_stndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        spatial_n_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=2,
        patience=2,
        mask_prob=0.25,
        mask_mode="block_time",
        span_length=2,
        heldin_loss_weight=0.1,
        validation_fraction=0.2,
        input_transform="sqrt_zscore",
        seed=0,
        contrast_loss_weight=0.0,
        ensemble_size=1,
        device="cpu",
    )

    assert out["train_rates_heldin"].shape == train_hi.shape
    assert out["eval_rates_heldout"].shape == (2, 8, 3)
    assert np.all(out["eval_rates_heldout"] > 0)


def test_fit_predict_stndt_lite_rejects_block_time_without_span_length() -> None:
    pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import fit_predict_stndt_lite

    rng = np.random.default_rng(502)
    train_hi = rng.poisson(0.5, (3, 8, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 8, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 8, 6)).astype(np.float32)

    with pytest.raises(ValueError, match="span_length"):
        fit_predict_stndt_lite(
            train_hi,
            train_ho,
            eval_hi,
            d_model=8,
            n_layers=1,
            n_heads=2,
            spatial_n_heads=2,
            dropout=0.0,
            learning_rate=0.003,
            weight_decay=0.0,
            batch_size=2,
            max_epochs=1,
            patience=1,
            mask_prob=0.25,
            mask_mode="block_time",
            span_length=0,
            heldin_loss_weight=0.1,
            validation_fraction=0.2,
            input_transform="sqrt_zscore",
            seed=0,
            contrast_loss_weight=0.0,
            ensemble_size=1,
            device="cpu",
        )


def test_fit_predict_stndt_lite_unit_calibration_identity_at_init() -> None:
    pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import fit_predict_stndt_lite

    rng = np.random.default_rng(504)
    train_hi = rng.poisson(0.5, (5, 8, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (5, 8, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (2, 8, 6)).astype(np.float32)

    out_off = fit_predict_stndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        spatial_n_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=1,
        patience=1,
        mask_prob=0.25,
        heldin_loss_weight=0.1,
        validation_fraction=0.2,
        input_transform="sqrt_zscore",
        seed=0,
        contrast_loss_weight=0.0,
        ensemble_size=1,
        unit_calibration=False,
        device="cpu",
    )
    out_on = fit_predict_stndt_lite(
        train_hi,
        train_ho,
        eval_hi,
        d_model=8,
        n_layers=1,
        n_heads=2,
        spatial_n_heads=2,
        dropout=0.0,
        learning_rate=0.003,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=1,
        patience=1,
        mask_prob=0.25,
        heldin_loss_weight=0.1,
        validation_fraction=0.2,
        input_transform="sqrt_zscore",
        seed=0,
        contrast_loss_weight=0.0,
        ensemble_size=1,
        unit_calibration=True,
        unit_calibration_scale_reg=10.0,
        unit_calibration_bias_reg=10.0,
        device="cpu",
    )

    assert out_off["eval_rates_heldout"].shape == out_on["eval_rates_heldout"].shape
    assert np.all(out_on["eval_rates_heldout"] > 0)


def test_fit_predict_stndt_lite_rejects_unknown_mask_mode() -> None:
    pytest.importorskip("torch")
    from nlb_project.models.stndt_lite import fit_predict_stndt_lite

    rng = np.random.default_rng(503)
    train_hi = rng.poisson(0.5, (3, 8, 6)).astype(np.float32)
    train_ho = rng.poisson(0.4, (3, 8, 3)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (1, 8, 6)).astype(np.float32)

    with pytest.raises(ValueError, match="mask_mode"):
        fit_predict_stndt_lite(
            train_hi,
            train_ho,
            eval_hi,
            d_model=8,
            n_layers=1,
            n_heads=2,
            spatial_n_heads=2,
            dropout=0.0,
            learning_rate=0.003,
            weight_decay=0.0,
            batch_size=2,
            max_epochs=1,
            patience=1,
            mask_prob=0.25,
            mask_mode="invalid",
            heldin_loss_weight=0.1,
            validation_fraction=0.2,
            input_transform="sqrt_zscore",
            seed=0,
            contrast_loss_weight=0.0,
            ensemble_size=1,
            device="cpu",
        )
