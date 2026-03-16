from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def _flatten_trial_time(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def predict_ridge_direct(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    """Predict held-out rates from held-in rates using direct multi-target ridge regression."""
    train_rates_heldin = np.asarray(train_rates_heldin, dtype=np.float32)
    train_rates_heldout = np.asarray(train_rates_heldout, dtype=np.float32)
    eval_rates_heldin = np.asarray(eval_rates_heldin, dtype=np.float32)

    n_train, tlen, _ = train_rates_heldin.shape
    n_eval = eval_rates_heldin.shape[0]
    n_ho = train_rates_heldout.shape[2]

    train_hi_2d = _flatten_trial_time(train_rates_heldin)
    train_ho_2d = _flatten_trial_time(train_rates_heldout)
    eval_hi_2d = _flatten_trial_time(eval_rates_heldin)

    ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
    ridge.fit(train_hi_2d, train_ho_2d)

    train_ho_pred = ridge.predict(train_hi_2d)
    eval_ho_pred = ridge.predict(eval_hi_2d)

    return {
        "train_rates_heldin": np.clip(train_rates_heldin, 1e-9, 1e20),
        "train_rates_heldout": np.clip(train_ho_pred.reshape(n_train, tlen, n_ho), 1e-9, 1e20),
        "eval_rates_heldin": np.clip(eval_rates_heldin, 1e-9, 1e20),
        "eval_rates_heldout": np.clip(eval_ho_pred.reshape(n_eval, tlen, n_ho), 1e-9, 1e20),
    }
