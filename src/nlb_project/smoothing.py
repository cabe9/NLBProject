from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor

logger = logging.getLogger(__name__)


@dataclass
class SmoothingParams:
    kern_sd_ms: float
    alpha: float
    log_offset: float


def _flatten_trial_time(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def _fit_poisson(
    train_factors: np.ndarray,
    eval_factors: np.ndarray,
    train_spikes: np.ndarray,
    eval_spikes: np.ndarray | None,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train_factors if eval_spikes is None else np.vstack([train_factors, eval_factors])
    y_train = train_spikes if eval_spikes is None else np.vstack([train_spikes, eval_spikes])

    train_pred = []
    eval_pred = []
    for chan in range(y_train.shape[1]):
        model = PoissonRegressor(alpha=alpha, max_iter=500)
        model.fit(x_train, y_train[:, chan])
        while model.n_iter_ == model.max_iter and model.max_iter < 10000:
            old = model.max_iter
            model = PoissonRegressor(alpha=alpha, max_iter=old * 2)
            model.fit(x_train, y_train[:, chan])
        train_pred.append(model.predict(train_factors))
        eval_pred.append(model.predict(eval_factors))

    train_rates = np.clip(np.vstack(train_pred).T, 1e-9, 1e20)
    eval_rates = np.clip(np.vstack(eval_pred).T, 1e-9, 1e20)
    return train_rates, eval_rates


def predict_rates(
    train_spikes_heldin: np.ndarray,
    train_spikes_heldout: np.ndarray,
    eval_spikes_heldin: np.ndarray,
    params: SmoothingParams,
    bin_size_ms: int,
) -> dict[str, np.ndarray]:
    # nlb_tools can return float16; SciPy gaussian_filter1d and sklearn can choke on it.
    train_spikes_heldin = np.asarray(train_spikes_heldin, dtype=np.float32)
    train_spikes_heldout = np.asarray(train_spikes_heldout, dtype=np.float32)
    eval_spikes_heldin = np.asarray(eval_spikes_heldin, dtype=np.float32)

    tlen = train_spikes_heldin.shape[1]
    n_heldout = train_spikes_heldout.shape[2]

    sigma_bins = max(1, int(params.kern_sd_ms / bin_size_ms))
    train_smoothed_heldin = gaussian_filter1d(train_spikes_heldin, sigma=sigma_bins, axis=1, mode="nearest")
    eval_smoothed_heldin = gaussian_filter1d(eval_spikes_heldin, sigma=sigma_bins, axis=1, mode="nearest")

    train_factors = np.log(_flatten_trial_time(train_smoothed_heldin) + params.log_offset)
    eval_factors = np.log(_flatten_trial_time(eval_smoothed_heldin) + params.log_offset)

    train_heldout_2d = _flatten_trial_time(train_spikes_heldout)
    train_rates_heldout_2d, eval_rates_heldout_2d = _fit_poisson(
        train_factors,
        eval_factors,
        train_heldout_2d,
        None,
        params.alpha,
    )

    return {
        "train_rates_heldin": train_smoothed_heldin,
        "train_rates_heldout": train_rates_heldout_2d.reshape(-1, tlen, n_heldout),
        "eval_rates_heldin": eval_smoothed_heldin,
        "eval_rates_heldout": eval_rates_heldout_2d.reshape(-1, tlen, n_heldout),
    }
