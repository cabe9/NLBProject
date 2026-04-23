"""Shape and param-sensitivity tests for the lagged-feature model family.

These guard that:

- each ``fit_predict_*`` returns tensors with the expected trial/time/neuron
  dimensions for both train and eval splits, and
- changing a hyperparameter (history, components, rank, ridge alpha) actually
  moves the eval predictions, so sweeps are doing real work rather than
  silently clamping to a default.

No numerical regression is pinned here -- those guarantees live in the
checked-in ``results/benchmark_runs/*/metrics.csv`` fixtures.
"""

import numpy as np

from nlb_project.models.lagged_pca_latent_regression import (
    fit_predict_lagged_pca_latent_regression,
)
from nlb_project.models.lagged_reduced_rank_regression import (
    fit_predict_lagged_reduced_rank_regression,
)
from nlb_project.models.lagged_ridge_direct import fit_predict_lagged_ridge_direct
from nlb_project.models.lds_pca_latent_regression import (
    fit_predict_lds_pca_latent_regression,
)


def test_lagged_ridge_shapes():
    """Lagged ridge returns train/eval tensors matching the held-out shape."""
    rng = np.random.default_rng(41)
    train_hi = rng.poisson(0.5, (6, 25, 8)).astype(float)
    train_ho = rng.poisson(0.5, (6, 25, 3)).astype(float)
    eval_hi = rng.poisson(0.5, (4, 25, 8)).astype(float)

    out = fit_predict_lagged_ridge_direct(
        train_hi,
        train_ho,
        eval_hi,
        ridge_alpha=0.1,
        history_bins=5,
        input_transform="sqrt",
    )

    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldout"].shape == (4, 25, 3)


def test_lagged_ridge_history_changes_output():
    """Changing ``history_bins`` perturbs eval predictions (sweep is effective)."""
    rng = np.random.default_rng(42)
    train_hi = rng.poisson(0.6, (6, 30, 10)).astype(float)
    train_ho = rng.poisson(0.6, (6, 30, 4)).astype(float)
    eval_hi = rng.poisson(0.6, (4, 30, 10)).astype(float)

    out_a = fit_predict_lagged_ridge_direct(
        train_hi, train_ho, eval_hi, ridge_alpha=0.1, history_bins=1, input_transform="sqrt"
    )
    out_b = fit_predict_lagged_ridge_direct(
        train_hi, train_ho, eval_hi, ridge_alpha=0.1, history_bins=5, input_transform="sqrt"
    )

    assert not np.allclose(out_a["eval_rates_heldout"], out_b["eval_rates_heldout"])


def test_lagged_pca_shapes():
    """Lagged PCA latent regression returns the expected tensor shapes."""
    rng = np.random.default_rng(43)
    train_hi = rng.poisson(0.5, (5, 20, 12)).astype(float)
    train_ho = rng.poisson(0.5, (5, 20, 4)).astype(float)
    eval_hi = rng.poisson(0.5, (3, 20, 12)).astype(float)

    out = fit_predict_lagged_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=6,
        ridge_alpha=0.1,
        history_bins=3,
        input_transform="sqrt_zscore",
    )

    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldout"].shape == (3, 20, 4)


def test_lagged_pca_params_change_output():
    """Distinct hyperparameters yield distinct lagged-PCA predictions."""
    rng = np.random.default_rng(44)
    train_hi = rng.poisson(0.7, (7, 25, 10)).astype(float)
    train_ho = rng.poisson(0.7, (7, 25, 4)).astype(float)
    eval_hi = rng.poisson(0.7, (4, 25, 10)).astype(float)

    out_a = fit_predict_lagged_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=5,
        ridge_alpha=0.01,
        history_bins=3,
        input_transform="sqrt_zscore",
    )
    out_b = fit_predict_lagged_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=12,
        ridge_alpha=1.0,
        history_bins=7,
        input_transform="sqrt_zscore",
    )

    assert not np.allclose(out_a["eval_rates_heldout"], out_b["eval_rates_heldout"])


def test_lagged_rrr_shapes():
    """Lagged reduced-rank regression returns the expected tensor shapes."""
    rng = np.random.default_rng(45)
    train_hi = rng.poisson(0.5, (6, 25, 12)).astype(float)
    train_ho = rng.poisson(0.5, (6, 25, 5)).astype(float)
    eval_hi = rng.poisson(0.5, (4, 25, 12)).astype(float)

    out = fit_predict_lagged_reduced_rank_regression(
        train_hi,
        train_ho,
        eval_hi,
        rank=3,
        ridge_alpha=0.1,
        history_bins=5,
        input_transform="sqrt_zscore",
    )

    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldout"].shape == (4, 25, 5)


def test_lagged_rrr_params_change_output():
    """Distinct rank/alpha/history yield distinct RRR predictions."""
    rng = np.random.default_rng(46)
    train_hi = rng.poisson(0.7, (8, 30, 10)).astype(float)
    train_ho = rng.poisson(0.7, (8, 30, 4)).astype(float)
    eval_hi = rng.poisson(0.7, (5, 30, 10)).astype(float)

    out_a = fit_predict_lagged_reduced_rank_regression(
        train_hi,
        train_ho,
        eval_hi,
        rank=2,
        ridge_alpha=0.01,
        history_bins=3,
        input_transform="sqrt_zscore",
    )
    out_b = fit_predict_lagged_reduced_rank_regression(
        train_hi,
        train_ho,
        eval_hi,
        rank=4,
        ridge_alpha=1.0,
        history_bins=7,
        input_transform="sqrt_zscore",
    )

    assert not np.allclose(out_a["eval_rates_heldout"], out_b["eval_rates_heldout"])


def test_lds_pca_shapes():
    """LDS-smoothed PCA regression returns the expected tensor shapes."""
    rng = np.random.default_rng(47)
    train_hi = rng.poisson(0.5, (6, 25, 12)).astype(float)
    train_ho = rng.poisson(0.5, (6, 25, 5)).astype(float)
    eval_hi = rng.poisson(0.5, (4, 25, 12)).astype(float)

    out = fit_predict_lds_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=6,
        ridge_alpha=0.1,
        input_transform="sqrt_zscore",
        obs_noise_scale=0.1,
    )

    assert out["train_rates_heldout"].shape == train_ho.shape
    assert out["eval_rates_heldout"].shape == (4, 25, 5)


def test_lds_pca_components_change_output():
    """Different ``n_components`` perturb LDS-PCA predictions."""
    rng = np.random.default_rng(48)
    train_hi = rng.poisson(0.7, (8, 30, 10)).astype(float)
    train_ho = rng.poisson(0.7, (8, 30, 4)).astype(float)
    eval_hi = rng.poisson(0.7, (5, 30, 10)).astype(float)

    out_a = fit_predict_lds_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=4,
        ridge_alpha=0.1,
        input_transform="sqrt_zscore",
        obs_noise_scale=0.1,
    )
    out_b = fit_predict_lds_pca_latent_regression(
        train_hi,
        train_ho,
        eval_hi,
        n_components=8,
        ridge_alpha=0.1,
        input_transform="sqrt_zscore",
        obs_noise_scale=0.1,
    )

    assert not np.allclose(out_a["eval_rates_heldout"], out_b["eval_rates_heldout"])
