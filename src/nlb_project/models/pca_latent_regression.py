from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from .output_head import OutputHead, fit_predict_rate_head

logger = logging.getLogger(__name__)


def _flatten_trial_time(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def predict_pca_latent_regression(
    train_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    *,
    n_components: int,
    ridge_alpha: float,
    output_head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Predict held-out rates from PCA latents of held-in activity.

    Inputs are trial x time x neuron arrays. Outputs match the tensor keys
    expected by ``nlb_tools.save_to_h5`` and ``nlb_tools.evaluation.evaluate``.

    The rate readout defaults to a log-link ridge so predictions are strictly
    positive and co-bps is not destroyed by clipped-Gaussian outputs. Pass
    ``output_head="linear"`` to reproduce the legacy behaviour for ablations.
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

    max_components = min(train_hi_2d.shape[0], train_hi_2d.shape[1])
    n_components_eff = max(1, min(int(n_components), max_components))
    if n_components_eff != int(n_components):
        logger.warning(
            "Requested n_components=%s exceeds allowed maximum=%s. Using n_components=%s.",
            n_components,
            max_components,
            n_components_eff,
        )

    pca = PCA(n_components=n_components_eff, svd_solver="auto", random_state=0)
    train_latent = pca.fit_transform(train_hi_2d)
    eval_latent = pca.transform(eval_hi_2d)

    train_pred_2d, eval_pred_2d = fit_predict_rate_head(
        train_latent,
        train_ho_2d,
        eval_latent,
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
