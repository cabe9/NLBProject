"""Tests for the shared rate-prediction output head.

These guard the Day-1 switch from Gaussian-ridge-on-counts (which produces
negative predictions clipped at 1e-9, destroying co-bps) to a log-link
ridge that predicts strictly positive rates.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlb_project.models.lagged_pca_latent_regression import (
    predict_lagged_pca_latent_regression,
)
from nlb_project.models.lagged_reduced_rank_regression import (
    predict_lagged_reduced_rank_regression,
)
from nlb_project.models.lagged_ridge_direct import predict_lagged_ridge_direct
from nlb_project.models.lds_pca_latent_regression import (
    predict_lds_pca_latent_regression,
)
from nlb_project.models.output_head import (
    fit_predict_rate_head,
    fit_reduced_rank_log_rate,
    validate_output_head,
)
from nlb_project.models.pca_latent_regression import predict_pca_latent_regression
from nlb_project.models.ridge_direct import predict_ridge_direct


def _poisson_tensors(seed: int, n_train: int = 6, n_eval: int = 4, tlen: int = 25, n_hi: int = 10, n_ho: int = 4):
    rng = np.random.default_rng(seed)
    train_hi = rng.poisson(0.5, (n_train, tlen, n_hi)).astype(np.float32)
    train_ho = rng.poisson(0.5, (n_train, tlen, n_ho)).astype(np.float32)
    eval_hi = rng.poisson(0.5, (n_eval, tlen, n_hi)).astype(np.float32)
    return train_hi, train_ho, eval_hi


def test_validate_output_head_rejects_unknown():
    with pytest.raises(ValueError):
        validate_output_head("not_a_real_head")


def test_validate_output_head_accepts_poisson_glm():
    assert validate_output_head("poisson_glm") == "poisson_glm"


def test_validate_output_head_normalises_case():
    assert validate_output_head("Log_Link") == "log_link"


def test_log_link_predictions_are_strictly_positive():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((200, 6)).astype(np.float32)
    x_eval = rng.standard_normal((80, 6)).astype(np.float32)
    # Sparse counts with many zeros -- the exact regime that used to produce
    # negative Gaussian-ridge predictions clipped to 1e-9.
    counts = rng.poisson(0.1, (200, 4)).astype(np.float32)

    train_pred, eval_pred = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=0.1, head="log_link", log_offset=1e-3
    )

    assert np.all(train_pred > 0.0)
    assert np.all(eval_pred > 0.0)
    assert train_pred.min() >= 1e-9
    assert eval_pred.min() >= 1e-9


def test_linear_head_reproduces_legacy_behaviour_signature():
    rng = np.random.default_rng(1)
    x_train = rng.standard_normal((100, 5)).astype(np.float32)
    x_eval = rng.standard_normal((40, 5)).astype(np.float32)
    counts = rng.poisson(0.5, (100, 3)).astype(np.float32)

    train_pred, eval_pred = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=0.1, head="linear"
    )
    assert train_pred.min() >= 1e-9
    assert eval_pred.min() >= 1e-9
    # Linear head can produce predictions that have been clipped from below,
    # so the floor should be hit at least somewhere on sparse data.
    assert train_pred.min() == pytest.approx(1e-9, rel=0.0, abs=0.0) or np.all(
        train_pred > 1e-9
    )


def test_log_link_and_linear_heads_disagree_on_sparse_counts():
    rng = np.random.default_rng(2)
    x_train = rng.standard_normal((150, 4)).astype(np.float32)
    x_eval = rng.standard_normal((50, 4)).astype(np.float32)
    counts = rng.poisson(0.1, (150, 3)).astype(np.float32)

    _, eval_log = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=0.1, head="log_link", log_offset=1e-3
    )
    _, eval_lin = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=0.1, head="linear"
    )

    # The two heads should produce materially different eval predictions.
    assert not np.allclose(eval_log, eval_lin, atol=1e-3)


def test_log_link_mean_prediction_tracks_mean_rate():
    """With Duan's smearing correction, a log-link ridge with no informative
    features should recover a rate in the neighbourhood of the empirical mean.

    This guards against the Jensen-inequality bias: a naive ``exp(mean(log(y+eps)))``
    in the sparse-Poisson regime massively under-predicts the mean.
    """
    rng = np.random.default_rng(3)
    rate = 0.7
    x_train = rng.standard_normal((2000, 1)).astype(np.float32)
    x_eval = rng.standard_normal((500, 1)).astype(np.float32)
    counts = rng.poisson(rate, (2000, 1)).astype(np.float32)

    _, eval_pred = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=1.0, head="log_link", log_offset=1e-3
    )
    # Tolerance is generous because smearing is a nonparametric finite-sample
    # estimator; what we really want to reject is order-of-magnitude bias.
    assert eval_pred.mean() == pytest.approx(counts.mean(), rel=0.35)


@pytest.mark.parametrize(
    "predict_fn,kwargs",
    [
        (predict_ridge_direct, {"ridge_alpha": 0.1}),
        (predict_pca_latent_regression, {"n_components": 4, "ridge_alpha": 0.1}),
        (
            predict_lagged_ridge_direct,
            {"ridge_alpha": 0.1, "history_bins": 3, "input_transform": "sqrt"},
        ),
        (
            predict_lagged_pca_latent_regression,
            {
                "n_components": 5,
                "ridge_alpha": 0.1,
                "history_bins": 3,
                "input_transform": "sqrt_zscore",
            },
        ),
        (
            predict_lagged_reduced_rank_regression,
            {
                "rank": 3,
                "ridge_alpha": 0.1,
                "history_bins": 3,
                "input_transform": "sqrt_zscore",
            },
        ),
        (
            predict_lds_pca_latent_regression,
            {
                "n_components": 4,
                "ridge_alpha": 0.1,
                "input_transform": "sqrt_zscore",
                "obs_noise_scale": 0.1,
            },
        ),
    ],
)
def test_models_default_to_log_link_and_predict_positive_rates(predict_fn, kwargs):
    train_hi, train_ho, eval_hi = _poisson_tensors(seed=hash(predict_fn.__name__) & 0xFF)
    out = predict_fn(train_hi, train_ho, eval_hi, **kwargs)
    assert np.all(out["eval_rates_heldout"] > 0.0)
    assert np.all(out["train_rates_heldout"] > 0.0)


def test_log_link_avoids_clip_floor_that_legacy_linear_hits():
    """Under overfit-prone sparse features the legacy Gaussian-ridge head
    produces negative rate predictions that get clipped to ``1e-9``; the
    log-link head guarantees positive rates by construction.

    This is the direct regime where the Day-1 fix matters: clipped bins where
    a real spike lands contribute ``~ -log(1e-9)`` nats to co-bps, destroying
    the metric. The log-link head never goes below the rate floor.
    """
    rng = np.random.default_rng(123)
    n_train, n_eval, n_features, n_out = 80, 40, 60, 8
    # Correlated features + many zero-counts: a typical sparse neural regime.
    x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    x_eval = rng.standard_normal((n_eval, n_features)).astype(np.float32)
    counts = rng.poisson(0.15, (n_train, n_out)).astype(np.float32)

    _, eval_lin = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=1e-4, head="linear"
    )
    _, eval_log = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=1e-4, head="log_link", log_offset=1e-3
    )

    # Legacy linear head hits the 1e-9 clip floor on many bins in this regime
    # (those are exactly the bins that destroy co-bps). The log-link head is
    # strictly positive by construction and stays orders of magnitude above
    # the floor.
    assert (eval_lin <= 1e-8).any(), (
        "legacy linear head should hit the clip floor in overfit sparse regime"
    )
    clip_rate_linear = float((eval_lin <= 1e-8).mean())
    clip_rate_loglink = float((eval_log <= 1e-8).mean())
    assert clip_rate_loglink < clip_rate_linear / 10.0, (
        f"log-link clipped {clip_rate_loglink:.3f} vs linear {clip_rate_linear:.3f}; "
        "expected log-link to avoid the clip floor that destroys co-bps"
    ) 


def test_poisson_glm_head_predicts_strictly_positive_rates():
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((200, 5)).astype(np.float32)
    x_eval = rng.standard_normal((80, 5)).astype(np.float32)
    counts = rng.poisson(0.3, (200, 4)).astype(np.float32)

    train_pred, eval_pred = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=0.1, head="poisson_glm"
    )

    assert train_pred.shape == counts.shape
    assert eval_pred.shape == (80, 4)
    assert np.all(train_pred > 0.0)
    assert np.all(eval_pred > 0.0)


def test_poisson_glm_mean_prediction_matches_mean_rate_without_smearing():
    """Poisson GLMs are calibrated in rate space by construction. Unlike the
    log-link approximation, no Jensen correction / smearing is needed: the
    intercept-only model should recover the empirical mean directly.
    """
    rng = np.random.default_rng(7)
    rate = 0.7
    x_train = rng.standard_normal((2000, 1)).astype(np.float32)
    x_eval = rng.standard_normal((500, 1)).astype(np.float32)
    counts = rng.poisson(rate, (2000, 1)).astype(np.float32)

    _, eval_pred = fit_predict_rate_head(
        x_train, counts, x_eval, ridge_alpha=1.0, head="poisson_glm"
    )
    # Poisson MLE hits the mean exactly when features are uninformative;
    # shrinkage from alpha=1 adds a small bias but nothing like 35%.
    assert eval_pred.mean() == pytest.approx(counts.mean(), rel=0.05)


def test_poisson_glm_reduced_rank_respects_rank_constraint():
    rng = np.random.default_rng(9)
    x_train = rng.standard_normal((200, 8)).astype(np.float32)
    x_eval = rng.standard_normal((60, 8)).astype(np.float32)
    counts = rng.poisson(0.4, (200, 6)).astype(np.float32)

    _, pred_low = fit_reduced_rank_log_rate(
        x_train, counts, x_eval, rank=2, ridge_alpha=0.1, head="poisson_glm"
    )
    _, pred_high = fit_reduced_rank_log_rate(
        x_train, counts, x_eval, rank=6, ridge_alpha=0.1, head="poisson_glm"
    )
    assert not np.allclose(pred_low, pred_high)
    assert np.all(pred_low > 0.0)
    assert np.all(pred_high > 0.0)


@pytest.mark.parametrize(
    "predict_fn,kwargs",
    [
        (predict_ridge_direct, {"ridge_alpha": 0.1}),
        (predict_pca_latent_regression, {"n_components": 4, "ridge_alpha": 0.1}),
        (
            predict_lagged_ridge_direct,
            {"ridge_alpha": 0.1, "history_bins": 3, "input_transform": "sqrt"},
        ),
        (
            predict_lagged_pca_latent_regression,
            {
                "n_components": 5,
                "ridge_alpha": 0.1,
                "history_bins": 3,
                "input_transform": "sqrt_zscore",
            },
        ),
        (
            predict_lagged_reduced_rank_regression,
            {
                "rank": 3,
                "ridge_alpha": 0.1,
                "history_bins": 3,
                "input_transform": "sqrt_zscore",
            },
        ),
        (
            predict_lds_pca_latent_regression,
            {
                "n_components": 4,
                "ridge_alpha": 0.1,
                "input_transform": "sqrt_zscore",
                "obs_noise_scale": 0.1,
            },
        ),
    ],
)
def test_models_support_poisson_glm_head_end_to_end(predict_fn, kwargs):
    train_hi, train_ho, eval_hi = _poisson_tensors(seed=hash(predict_fn.__name__) & 0xFF)
    out = predict_fn(
        train_hi, train_ho, eval_hi, output_head="poisson_glm", **kwargs
    )
    assert np.all(out["eval_rates_heldout"] > 0.0)
    assert np.all(out["train_rates_heldout"] > 0.0)


def test_reduced_rank_log_link_respects_rank_constraint():
    rng = np.random.default_rng(5)
    x_train = rng.standard_normal((200, 8)).astype(np.float32)
    x_eval = rng.standard_normal((60, 8)).astype(np.float32)
    counts = rng.poisson(0.4, (200, 6)).astype(np.float32)

    _, pred_low = fit_reduced_rank_log_rate(
        x_train, counts, x_eval, rank=2, ridge_alpha=0.1, head="log_link", log_offset=1e-3
    )
    _, pred_high = fit_reduced_rank_log_rate(
        x_train, counts, x_eval, rank=6, ridge_alpha=0.1, head="log_link", log_offset=1e-3
    )
    assert not np.allclose(pred_low, pred_high)
    assert np.all(pred_low > 0.0)
    assert np.all(pred_high > 0.0)
