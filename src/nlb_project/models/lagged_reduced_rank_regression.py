from __future__ import annotations

import logging

import numpy as np

from .output_head import OutputHead, fit_reduced_rank_log_rate
from .temporal_features import _flatten_trial_time, apply_input_transform, build_history_features

logger = logging.getLogger(__name__)


def fit_predict_lagged_reduced_rank_regression(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    rank: int,
    ridge_alpha: float,
    history_bins: int,
    input_transform: str = "sqrt_zscore",
    output_head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Predict held-out rates with lagged reduced-rank regression.

    Under ``output_head="log_link"`` (default) the rank constraint is applied
    in log-rate space: the ridge targets are ``log(count + log_offset)`` and
    the rank-``r`` coefficient is projected onto the top response subspace
    before exponentiation. This keeps the low-rank structure of the original
    model while producing strictly positive rate predictions.
    """
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

    train_pred_2d, eval_pred_2d = fit_reduced_rank_log_rate(
        train_x,
        train_y,
        eval_x,
        rank=rank,
        ridge_alpha=ridge_alpha,
        head=output_head,
        log_offset=log_offset,
    )

    return {
        "train_rates_heldin": np.clip(train_rates_heldin, 1e-9, 1e20),
        "train_rates_heldout": train_pred_2d.reshape(n_train, tlen, n_ho),
        "eval_rates_heldin": np.clip(eval_rates_heldin, 1e-9, 1e20),
        "eval_rates_heldout": eval_pred_2d.reshape(n_eval, tlen, n_ho),
    }
