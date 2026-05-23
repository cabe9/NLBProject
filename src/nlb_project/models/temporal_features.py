from __future__ import annotations

import numpy as np


def _flatten_trial_time(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, arr.shape[2])


def build_history_features(arr: np.ndarray, history_bins: int) -> np.ndarray:
    """Build trial-safe lagged features with zero padding at trial start.

    Output shape: trials x time x (neurons * history_bins)
    The first block is the current bin, followed by 1-step-back, etc.
    """
    arr = np.asarray(arr, dtype=np.float32)
    history_bins = max(1, int(history_bins))

    features = [arr]
    for lag in range(1, history_bins):
        shifted = np.zeros_like(arr)
        shifted[:, lag:, :] = arr[:, :-lag, :]
        features.append(shifted)
    return np.concatenate(features, axis=2)


def apply_input_transform(
    train_2d: np.ndarray,
    eval_2d: np.ndarray,
    *,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply train-fit-only transforms to design matrices."""
    transform = transform.lower()
    train_out = np.asarray(train_2d, dtype=np.float32)
    eval_out = np.asarray(eval_2d, dtype=np.float32)

    if transform in {"sqrt", "sqrt_zscore"}:
        train_out = np.sqrt(np.clip(train_out, 0.0, None))
        eval_out = np.sqrt(np.clip(eval_out, 0.0, None))

    if transform in {"zscore", "sqrt_zscore"}:
        mean = train_out.mean(axis=0, keepdims=True)
        std = train_out.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        train_out = (train_out - mean) / std
        eval_out = (eval_out - mean) / std

    if transform not in {"none", "sqrt", "zscore", "sqrt_zscore"}:
        raise ValueError(
            f"Unsupported input transform `{transform}`. "
            "Expected one of ['none', 'sqrt', 'zscore', 'sqrt_zscore']."
        )

    return train_out, eval_out
