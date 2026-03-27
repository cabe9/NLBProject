from __future__ import annotations

import logging

import numpy as np

from .temporal_features import _flatten_trial_time, apply_input_transform, build_history_features

logger = logging.getLogger(__name__)


def _fit_reduced_rank_weights(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    rank: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ridge-regularized reduced-rank regression with train-only centering.

    The rank constraint is applied to the fitted response subspace:
    B_rrr = B_ridge V_r V_r^T
    where V_r are the top right singular vectors of X B_ridge.
    """
    x_mean = train_x.mean(axis=0, keepdims=True)
    y_mean = train_y.mean(axis=0, keepdims=True)
    xc = train_x - x_mean
    yc = train_y - y_mean

    xtx = xc.T @ xc
    reg = float(ridge_alpha) * np.eye(xtx.shape[0], dtype=np.float32)
    b_ridge = np.linalg.solve(xtx + reg, xc.T @ yc)

    y_hat = xc @ b_ridge
    _, _, vt = np.linalg.svd(y_hat, full_matrices=False)
    max_rank = min(vt.shape[0], yc.shape[1])
    rank_eff = max(1, min(int(rank), max_rank))
    if rank_eff != int(rank):
        logger.warning(
            "Requested rank=%s exceeds allowed maximum=%s. Using rank=%s.",
            rank,
            max_rank,
            rank_eff,
        )
    v_r = vt[:rank_eff].T
    b_rrr = b_ridge @ v_r @ v_r.T
    intercept = y_mean - x_mean @ b_rrr
    return b_rrr.astype(np.float32), intercept.astype(np.float32)


def predict_lagged_reduced_rank_regression(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    rank: int,
    ridge_alpha: float,
    history_bins: int,
    input_transform: str = "sqrt_zscore",
) -> dict[str, np.ndarray]:
    """Predict held-out rates with lagged reduced-rank regression."""
    train_rates_heldin = np.asarray(train_rates_heldin, dtype=np.float32)
    train_rates_heldout = np.asarray(train_rates_heldout, dtype=np.float32)
    eval_rates_heldin = np.asarray(eval_rates_heldin, dtype=np.float32)

    n_train, tlen, _ = train_rates_heldin.shape
    n_eval = eval_rates_heldin.shape[0]
    n_ho = train_rates_heldout.shape[2]

    train_hist = build_history_features(train_rates_heldin, history_bins)
    eval_hist = build_history_features(eval_rates_heldin, history_bins)

    train_x = _flatten_trial_time(train_hist)
    eval_x = _flatten_trial_time(eval_hist)
    train_x, eval_x = apply_input_transform(train_x, eval_x, transform=input_transform)
    train_y = _flatten_trial_time(train_rates_heldout)

    weights, intercept = _fit_reduced_rank_weights(
        train_x,
        train_y,
        rank=rank,
        ridge_alpha=ridge_alpha,
    )

    train_pred = (train_x @ weights + intercept).reshape(n_train, tlen, n_ho)
    eval_pred = (eval_x @ weights + intercept).reshape(n_eval, tlen, n_ho)

    return {
        "train_rates_heldin": np.clip(train_rates_heldin, 1e-9, 1e20),
        "train_rates_heldout": np.clip(train_pred, 1e-9, 1e20),
        "eval_rates_heldin": np.clip(eval_rates_heldin, 1e-9, 1e20),
        "eval_rates_heldout": np.clip(eval_pred, 1e-9, 1e20),
    }
