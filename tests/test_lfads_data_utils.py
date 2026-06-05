"""Tests for LFADS HDF5 helpers (no lfads-torch import required)."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lfads_data_utils import (
    analyze_lfads_metrics_csv,
    assert_finite_h5,
    inspect_h5,
    lfads_training_overrides,
    subset_h5,
)


def _write_tiny_h5(path: Path, n_train: int = 10, n_valid: int = 4) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset(
            "train_encod_data",
            data=rng.random((n_train, 5, 3), dtype=np.float32).astype(np.float16),
        )
        h5file.create_dataset(
            "train_recon_data",
            data=rng.random((n_train, 7, 4), dtype=np.float32).astype(np.float16),
        )
        h5file.create_dataset(
            "valid_encod_data",
            data=rng.random((n_valid, 5, 3), dtype=np.float32).astype(np.float16),
        )
        h5file.create_dataset(
            "valid_recon_data",
            data=rng.random((n_valid, 7, 4), dtype=np.float32).astype(np.float16),
        )


def test_subset_h5_reduces_trials(tmp_path: Path) -> None:
    src = tmp_path / "full.h5"
    dst = tmp_path / "smoke.h5"
    _write_tiny_h5(src, n_train=10, n_valid=6)
    shapes = subset_h5(src, dst, max_train_trials=3, max_valid_trials=2)
    assert shapes["train_encod_data"] == [3, 5, 3]
    assert shapes["valid_encod_data"] == [2, 5, 3]
    assert_finite_h5(dst)


def test_inspect_h5_keys(tmp_path: Path) -> None:
    src = tmp_path / "full.h5"
    _write_tiny_h5(src)
    shapes = inspect_h5(src)
    assert "train_recon_data" in shapes
    assert shapes["train_recon_data"] == [10, 7, 4]


def test_lfads_training_overrides_stability_knobs(tmp_path: Path) -> None:
    src = tmp_path / "full.h5"
    _write_tiny_h5(src)
    overrides = lfads_training_overrides(
        src,
        batch_size=8,
        max_epochs=30,
        seed=0,
        lr_init=1e-3,
        gradient_clip_val=1.0,
    )
    assert overrides["datamodule.batch_size"] == 8
    assert overrides["trainer.max_epochs"] == 30
    assert overrides["model.lr_init"] == 1e-3
    assert overrides["trainer.gradient_clip_val"] == 1.0


def test_analyze_lfads_metrics_csv_detects_divergence(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,valid/recon_smth,valid/loss\n"
        "0,0.5,1.0\n"
        "1,0.4,0.9\n"
        "2,nan,nan\n",
        encoding="utf-8",
    )
    summary = analyze_lfads_metrics_csv(metrics)
    assert summary["available"] is True
    assert summary["diverged"] is True
    assert summary["first_nan_epoch"] == 2
    assert summary["last_finite_epoch"] == 1
    assert summary["best_epoch"] == 1
