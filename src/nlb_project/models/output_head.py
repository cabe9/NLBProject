"""Shared output-head utilities for rate prediction.

Every model in this repo maps features to held-out firing rates. For co-bps
(a Poisson log-likelihood metric) the rate predictions must be strictly
positive; clipping Gaussian-ridge predictions at a floor like 1e-9 wrecks the
metric whenever a spike lands in a near-zero-prediction bin.

This module centralises the rate readout so every model can share the same,
correctly-shaped head. Three modes are supported:

``"linear"``
    Legacy behaviour. Fit ``Ridge`` on raw spike counts, then clip predictions
    to ``[1e-9, 1e20]``. Kept for backward compatibility and ablation only.

``"log_link"`` (default)
    Fit ``Ridge`` on ``log(count + log_offset)`` and exponentiate at inference
    with Duan's smearing correction. Strictly-positive rates, and much faster
    than ``poisson_glm``. The default because it is fast enough for full CV
    sweeps while avoiding the ``1e-9`` clip-floor pathology of ``"linear"``.

``"poisson_glm"``
    Per-neuron ``sklearn.linear_model.PoissonRegressor`` (log-link Poisson
    IRLS). The co-bps-correct readout: the model is fit under the same
    likelihood that the metric scores. ~10-100x slower than the two Gaussian
    heads but removes the Jensen-bias / smearing approximation of
    ``log_link``. Opt in per model via the ``output_head`` config key.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
from sklearn.linear_model import PoissonRegressor, Ridge

logger = logging.getLogger(__name__)

OutputHead = Literal["linear", "log_link", "poisson_glm"]

_VALID_HEADS: tuple[str, ...] = ("linear", "log_link", "poisson_glm")
_RATE_FLOOR: float = 1e-9
_RATE_CEIL: float = 1e20
_POISSON_INITIAL_MAX_ITER: int = 500
_POISSON_ABSOLUTE_MAX_ITER: int = 10000


def validate_output_head(head: str) -> OutputHead:
    head_lower = str(head).lower()
    if head_lower not in _VALID_HEADS:
        raise ValueError(f"Unsupported output head `{head}`. Expected one of {_VALID_HEADS}.")
    return cast(OutputHead, head_lower)


def _prepare_log_targets(train_counts: np.ndarray, log_offset: float) -> np.ndarray:
    if log_offset <= 0.0:
        raise ValueError(f"log_offset must be > 0 for log_link head, got {log_offset}.")
    return np.log(train_counts.astype(np.float32) + np.float32(log_offset))


def _as_2d(arr: np.ndarray, n_outputs: int) -> np.ndarray:
    """Ensure a prediction array is ``(n_samples, n_outputs)``.

    ``sklearn.linear_model.Ridge.predict`` drops the trailing dim when
    ``n_targets == 1``, which silently broadcasts into ``(n_samples, n_samples)``
    residuals downstream. Forcing 2D shape here kills that bug class.
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.shape[-1] != n_outputs:
        raise ValueError(f"Expected prediction with {n_outputs} outputs, got shape {arr.shape}.")
    return arr


def _inverse_log_link(
    pred: np.ndarray, log_offset: float, smear: np.ndarray | None = None
) -> np.ndarray:
    """Map a log-link prediction back to the rate scale.

    The target space is ``log(count + log_offset)`` so ``exp(pred) * smear`` is
    an estimator of ``E[count + log_offset | x]`` (Duan's smearing corrects
    the Jensen bias that would otherwise make ``exp(E[log(y)])`` under-predict
    the mean). We use that quantity directly as the rate prediction rather
    than subtracting ``log_offset`` back: under co-bps the resulting
    ``+log_offset`` bias is negligible, but it guarantees predictions stay
    strictly positive instead of being driven into the ``1e-9`` clip floor
    whenever the model wants to predict "near zero". The floor that the old
    Gaussian-ridge head kept hitting was exactly what destroyed co-bps.
    """
    factor = np.exp(pred)
    if smear is not None:
        factor = factor * smear[None, :]
    return np.clip(factor, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)


def _fit_smearing(residuals: np.ndarray) -> np.ndarray:
    """Per-output Duan's smearing factor ``E[exp(residual)]`` on train data."""
    # Clip the residuals very conservatively to guard against numerical blow-up
    # when a near-zero bin sees a very negative residual; 50 is well beyond any
    # realistic log-count range in these datasets.
    clipped = np.clip(residuals, -50.0, 50.0)
    return np.exp(clipped).mean(axis=0).astype(np.float32)


def _fit_poisson_glm_single(
    train_features: np.ndarray,
    train_counts_col: np.ndarray,
    *,
    alpha: float,
) -> PoissonRegressor:
    """Fit one ``PoissonRegressor`` with adaptive max_iter.

    Matches the convergence-retry policy of :mod:`nlb_project.smoothing`:
    if the L-BFGS solver hits ``max_iter`` without converging, double the
    budget and retry up to ``_POISSON_ABSOLUTE_MAX_ITER``. This is a minor
    but important detail - on sparse spike counts the default ``max_iter=100``
    frequently returns non-converged solutions that silently degrade co-bps.
    """
    model = PoissonRegressor(alpha=float(alpha), max_iter=_POISSON_INITIAL_MAX_ITER)
    model.fit(train_features, train_counts_col)
    while model.n_iter_ == model.max_iter and model.max_iter < _POISSON_ABSOLUTE_MAX_ITER:
        model = PoissonRegressor(alpha=float(alpha), max_iter=model.max_iter * 2)
        model.fit(train_features, train_counts_col)
    return model


def _fit_per_neuron_poisson(
    train_features: np.ndarray,
    train_counts: np.ndarray,
    eval_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one ``PoissonRegressor`` per output neuron.

    Returns train/eval rate predictions along with the stacked coefficient
    matrix ``(n_features, n_outputs)`` and intercepts ``(n_outputs,)`` so
    downstream code (e.g. reduced-rank-Poisson) can rank-truncate.
    """
    n_features = train_features.shape[1]
    n_outputs = train_counts.shape[1]
    coef = np.zeros((n_features, n_outputs), dtype=np.float32)
    intercept = np.zeros(n_outputs, dtype=np.float32)
    train_pred = np.empty_like(train_counts, dtype=np.float32)
    eval_pred = np.empty((eval_features.shape[0], n_outputs), dtype=np.float32)
    for c in range(n_outputs):
        model = _fit_poisson_glm_single(train_features, train_counts[:, c], alpha=alpha)
        coef[:, c] = model.coef_.astype(np.float32)
        intercept[c] = float(model.intercept_)
        train_pred[:, c] = model.predict(train_features).astype(np.float32)
        eval_pred[:, c] = model.predict(eval_features).astype(np.float32)
    train_pred = np.clip(train_pred, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
    eval_pred = np.clip(eval_pred, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
    return train_pred, eval_pred, coef, intercept


def fit_predict_rate_head(
    train_features: np.ndarray,
    train_counts: np.ndarray,
    eval_features: np.ndarray,
    *,
    ridge_alpha: float,
    head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a readout and return nonnegative rate predictions.

    Parameters
    ----------
    train_features, eval_features : 2D float arrays
        Design matrices of shape ``(n_samples, n_features)``.
    train_counts : 2D float array
        Raw held-out spike counts per bin, shape ``(n_samples, n_outputs)``.
    ridge_alpha : float
        L2 penalty. Passed to ``Ridge`` for ``"linear"`` / ``"log_link"``
        and to ``PoissonRegressor`` for ``"poisson_glm"``.
    head : {"linear", "log_link", "poisson_glm"}
        Output head mode. Default is ``"log_link"`` (Gaussian ridge on
        log-counts with Duan's smearing), which is fast and strictly positive.
        ``"poisson_glm"`` is the co-bps-correct readout (per-neuron
        ``PoissonRegressor``) but is ~10-100x slower. ``"linear"`` reproduces
        the legacy Gaussian-ridge-on-counts behaviour and is kept for ablation.
    log_offset : float
        Offset inside the log link. Only used when ``head="log_link"``.

    Returns
    -------
    train_pred, eval_pred : np.ndarray
        Strictly positive rate predictions, shape ``(n_samples, n_outputs)``.
    """
    head = validate_output_head(head)
    train_features = np.asarray(train_features, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    train_counts = np.asarray(train_counts, dtype=np.float32)

    n_outputs = train_counts.shape[1]

    if head == "linear":
        ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
        ridge.fit(train_features, train_counts)
        train_raw = _as_2d(ridge.predict(train_features), n_outputs)
        eval_raw = _as_2d(ridge.predict(eval_features), n_outputs)
        train_pred = np.clip(train_raw, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        eval_pred = np.clip(eval_raw, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        return train_pred, eval_pred

    if head == "log_link":
        ridge = Ridge(alpha=float(ridge_alpha), random_state=0)
        train_targets = _prepare_log_targets(train_counts, log_offset)
        ridge.fit(train_features, train_targets)
        train_log_pred = _as_2d(ridge.predict(train_features), n_outputs)
        eval_log_pred = _as_2d(ridge.predict(eval_features), n_outputs)
        smear = _fit_smearing(train_targets - train_log_pred)
        train_pred = _inverse_log_link(train_log_pred, log_offset, smear)
        eval_pred = _inverse_log_link(eval_log_pred, log_offset, smear)
        return train_pred, eval_pred

    # head == "poisson_glm"
    train_pred, eval_pred, _, _ = _fit_per_neuron_poisson(
        train_features, train_counts, eval_features, alpha=ridge_alpha
    )
    return train_pred, eval_pred


def _rank_truncate(coef: np.ndarray, rank: int) -> np.ndarray:
    """Rank-``r`` SVD truncation of a ``(d, N)`` coefficient matrix.

    Matches the structural rank constraint used by the legacy Gaussian RRR
    path (response-subspace SVD projection) so comparisons across output
    heads measure like-for-like model capacity.
    """
    u, s, vt = np.linalg.svd(coef, full_matrices=False)
    max_rank = len(s)
    rank_eff = max(1, min(int(rank), max_rank))
    if rank_eff != int(rank):
        logger.warning(
            "Requested rank=%s exceeds allowed maximum=%s. Using rank=%s.",
            rank,
            max_rank,
            rank_eff,
        )
    return (u[:, :rank_eff] * s[:rank_eff]) @ vt[:rank_eff]


def fit_reduced_rank_log_rate(
    train_features: np.ndarray,
    train_counts: np.ndarray,
    eval_features: np.ndarray,
    *,
    rank: int,
    ridge_alpha: float,
    head: OutputHead = "log_link",
    log_offset: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a rank-constrained readout in rate space and return rates.

    The rank constraint is applied to the fitted coefficient matrix via SVD
    truncation. Under ``"linear"`` this reproduces the legacy Gaussian RRR.
    Under ``"log_link"`` and ``"poisson_glm"`` the rank-r structure lives in
    log-rate space, which is the natural space for a Poisson readout.
    """
    head = validate_output_head(head)
    train_features = np.asarray(train_features, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    train_counts = np.asarray(train_counts, dtype=np.float32)

    if head == "poisson_glm":
        # Fit full-rank per-neuron Poisson GLMs, then rank-truncate the stacked
        # coefficient matrix via SVD. The intercept is preserved.
        _, _, coef, intercept = _fit_per_neuron_poisson(
            train_features, train_counts, eval_features, alpha=ridge_alpha
        )
        coef_r = _rank_truncate(coef, rank)
        train_log = train_features @ coef_r + intercept
        eval_log = eval_features @ coef_r + intercept
        train_pred = np.clip(np.exp(train_log), _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        eval_pred = np.clip(np.exp(eval_log), _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        return train_pred, eval_pred

    if head == "linear":
        targets = train_counts
    else:
        targets = _prepare_log_targets(train_counts, log_offset)

    x_mean = train_features.mean(axis=0, keepdims=True)
    y_mean = targets.mean(axis=0, keepdims=True)
    xc = train_features - x_mean
    yc = targets - y_mean

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

    train_lin = (train_features @ b_rrr + intercept).astype(np.float32)
    eval_lin = (eval_features @ b_rrr + intercept).astype(np.float32)

    if head == "linear":
        train_pred = np.clip(train_lin, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        eval_pred = np.clip(eval_lin, _RATE_FLOOR, _RATE_CEIL).astype(np.float32)
        return train_pred, eval_pred

    # log_link: Duan's smearing against training residuals in log-rate space.
    smear = _fit_smearing(targets - train_lin)
    train_pred = _inverse_log_link(train_lin, log_offset, smear)
    eval_pred = _inverse_log_link(eval_lin, log_offset, smear)
    return train_pred, eval_pred
