from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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
) -> dict[str, np.ndarray]:
    """Predict held-out rates from PCA latents of held-in activity.

    Inputs are trial x time x neuron arrays. Outputs match the tensor keys expected
    by nlb_tools.save_to_h5 and nlb_tools.evaluation.evaluate.
    """
    train_rates_heldin = np.asarray(train_rates_heldin, dtype=np.float32)
    train_rates_heldout = np.asarray(train_rates_heldout, dtype=np.float32)
    eval_rates_heldin = np.asarray(eval_rates_heldin, dtype=np.float32)

    n_train, tlen, n_hi = train_rates_heldin.shape
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

    ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
    ridge.fit(train_latent, train_ho_2d)
    train_ho_pred = ridge.predict(train_latent)
    eval_ho_pred = ridge.predict(eval_latent)

    return {
        "train_rates_heldin": np.clip(train_rates_heldin, 1e-9, 1e20),
        "train_rates_heldout": np.clip(train_ho_pred.reshape(n_train, tlen, n_ho), 1e-9, 1e20),
        "eval_rates_heldin": np.clip(eval_rates_heldin, 1e-9, 1e20),
        "eval_rates_heldout": np.clip(eval_ho_pred.reshape(n_eval, tlen, n_ho), 1e-9, 1e20),
    }
