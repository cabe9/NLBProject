from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from .output_head import OutputHead, fit_predict_rate_head
from .temporal_features import _flatten_trial_time, apply_input_transform

logger = logging.getLogger(__name__)


def _fit_diag_lds_params(latents_3d: np.ndarray, obs_noise_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a diagonal AR(1) latent model from train-only PCA latents.

    The PCA latents are treated as noisy observations of a smoother latent state:
        z_t = a * z_{t-1} + w_t
        y_t = z_t + v_t
    where each latent dimension is modeled independently.
    """
    y_prev = latents_3d[:, :-1, :].reshape(-1, latents_3d.shape[2])
    y_next = latents_3d[:, 1:, :].reshape(-1, latents_3d.shape[2])

    denom = np.sum(y_prev * y_prev, axis=0)
    denom[denom < 1e-6] = 1e-6
    a = np.sum(y_prev * y_next, axis=0) / denom
    a = np.clip(a, -0.99, 0.99).astype(np.float32)

    resid = y_next - y_prev * a[None, :]
    q = np.var(resid, axis=0).astype(np.float32)
    latent_var = np.var(latents_3d.reshape(-1, latents_3d.shape[2]), axis=0).astype(np.float32)
    q = np.maximum(q, 1e-4)
    r = np.maximum(latent_var * float(obs_noise_scale), 1e-4).astype(np.float32)
    p0 = np.maximum(latent_var + q, 1e-4).astype(np.float32)
    return a, q, r, p0


def _smooth_trial_diag_lds(obs: np.ndarray, a: np.ndarray, q: np.ndarray, r: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Run diagonal Kalman filtering + RTS smoothing on one trial."""
    tlen, dim = obs.shape
    filt_mean = np.zeros((tlen, dim), dtype=np.float32)
    filt_var = np.zeros((tlen, dim), dtype=np.float32)
    pred_mean = np.zeros((tlen, dim), dtype=np.float32)
    pred_var = np.zeros((tlen, dim), dtype=np.float32)

    pred_mean[0] = 0.0
    pred_var[0] = p0
    gain0 = pred_var[0] / (pred_var[0] + r)
    filt_mean[0] = pred_mean[0] + gain0 * (obs[0] - pred_mean[0])
    filt_var[0] = (1.0 - gain0) * pred_var[0]

    for t in range(1, tlen):
        pred_mean[t] = a * filt_mean[t - 1]
        pred_var[t] = a * a * filt_var[t - 1] + q
        gain = pred_var[t] / (pred_var[t] + r)
        filt_mean[t] = pred_mean[t] + gain * (obs[t] - pred_mean[t])
        filt_var[t] = (1.0 - gain) * pred_var[t]

    smooth_mean = filt_mean.copy()
    smooth_var = filt_var.copy()
    for t in range(tlen - 2, -1, -1):
        denom = pred_var[t + 1].copy()
        denom[denom < 1e-6] = 1e-6
        smoother_gain = filt_var[t] * a / denom
        smooth_mean[t] = filt_mean[t] + smoother_gain * (smooth_mean[t + 1] - pred_mean[t + 1])
        smooth_var[t] = filt_var[t] + smoother_gain * smoother_gain * (smooth_var[t + 1] - pred_var[t + 1])

    return smooth_mean


def _smooth_latents(latents_3d: np.ndarray, a: np.ndarray, q: np.ndarray, r: np.ndarray, p0: np.ndarray) -> np.ndarray:
    out = np.zeros_like(latents_3d, dtype=np.float32)
    for trial_idx in range(latents_3d.shape[0]):
        out[trial_idx] = _smooth_trial_diag_lds(latents_3d[trial_idx], a, q, r, p0)
    return out


def predict_lds_pca_latent_regression(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    n_components: int,
    ridge_alpha: float,
    input_transform: str = "sqrt_zscore",
    obs_noise_scale: float = 0.1,
    output_head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Predict held-out rates from PCA latents smoothed by a diagonal Gaussian LDS.

    The rate readout defaults to a log-link ridge so predictions are strictly
    positive. Pass ``output_head="linear"`` for the legacy Gaussian readout.
    """
    train_rates_heldin = np.asarray(train_rates_heldin, dtype=np.float32)
    train_rates_heldout = np.asarray(train_rates_heldout, dtype=np.float32)
    eval_rates_heldin = np.asarray(eval_rates_heldin, dtype=np.float32)

    n_train, tlen, _ = train_rates_heldin.shape
    n_eval = eval_rates_heldin.shape[0]
    n_ho = train_rates_heldout.shape[2]

    train_x = _flatten_trial_time(train_rates_heldin)
    eval_x = _flatten_trial_time(eval_rates_heldin)
    train_x, eval_x = apply_input_transform(train_x, eval_x, transform=input_transform)
    train_y = _flatten_trial_time(train_rates_heldout)

    max_components = min(train_x.shape[0], train_x.shape[1])
    n_components_eff = max(1, min(int(n_components), max_components))
    if n_components_eff != int(n_components):
        logger.warning(
            "Requested n_components=%s exceeds allowed maximum=%s. Using n_components=%s.",
            n_components,
            max_components,
            n_components_eff,
        )

    pca = PCA(n_components=n_components_eff, svd_solver="auto", random_state=0)
    train_lat_obs = pca.fit_transform(train_x).reshape(n_train, tlen, n_components_eff).astype(np.float32)
    eval_lat_obs = pca.transform(eval_x).reshape(n_eval, tlen, n_components_eff).astype(np.float32)

    a, q, r, p0 = _fit_diag_lds_params(train_lat_obs, obs_noise_scale=float(obs_noise_scale))
    train_lat_smooth = _smooth_latents(train_lat_obs, a, q, r, p0).reshape(-1, n_components_eff)
    eval_lat_smooth = _smooth_latents(eval_lat_obs, a, q, r, p0).reshape(-1, n_components_eff)

    train_pred_2d, eval_pred_2d = fit_predict_rate_head(
        train_lat_smooth,
        train_y,
        eval_lat_smooth,
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
