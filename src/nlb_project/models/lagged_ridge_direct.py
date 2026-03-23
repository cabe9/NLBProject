from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .temporal_features import _flatten_trial_time, apply_input_transform, build_history_features


def predict_lagged_ridge_direct(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    ridge_alpha: float,
    history_bins: int,
    input_transform: str = "sqrt",
) -> dict[str, np.ndarray]:
    """Predict held-out rates from lagged held-in features using multi-target ridge."""
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

    ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
    ridge.fit(train_x, train_y)

    train_pred = ridge.predict(train_x).reshape(n_train, tlen, n_ho)
    eval_pred = ridge.predict(eval_x).reshape(n_eval, tlen, n_ho)

    return {
        "train_rates_heldin": np.clip(train_rates_heldin, 1e-9, 1e20),
        "train_rates_heldout": np.clip(train_pred, 1e-9, 1e20),
        "eval_rates_heldin": np.clip(eval_rates_heldin, 1e-9, 1e20),
        "eval_rates_heldout": np.clip(eval_pred, 1e-9, 1e20),
    }
