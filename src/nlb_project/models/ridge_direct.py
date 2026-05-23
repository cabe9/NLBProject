from __future__ import annotations

import numpy as np

from .output_head import OutputHead, fit_predict_rate_head


def _flatten_trial_time(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def fit_predict_ridge_direct(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    ridge_alpha: float,
    output_head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Predict held-out rates from held-in rates with direct multi-target ridge.

    By default the readout is a log-link ridge (regress on ``log(count +
    log_offset)``, exponentiate at inference). Pass ``output_head="linear"``
    to recover the legacy Gaussian-ridge-on-counts behaviour.
    """
    train_rates_heldin = np.asarray(train_rates_heldin, dtype=np.float32)
    train_rates_heldout = np.asarray(train_rates_heldout, dtype=np.float32)
    eval_rates_heldin = np.asarray(eval_rates_heldin, dtype=np.float32)

    n_train, tlen, _ = train_rates_heldin.shape
    n_eval = eval_rates_heldin.shape[0]
    n_ho = train_rates_heldout.shape[2]

    train_hi_2d = _flatten_trial_time(train_rates_heldin)
    train_ho_2d = _flatten_trial_time(train_rates_heldout)
    eval_hi_2d = _flatten_trial_time(eval_rates_heldin)

    train_pred_2d, eval_pred_2d = fit_predict_rate_head(
        train_hi_2d,
        train_ho_2d,
        eval_hi_2d,
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
