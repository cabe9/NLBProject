"""Tests for LFADS HDF5 helpers (no lfads-torch import required)."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lfads_data_utils import assert_finite_h5, inspect_h5, subset_h5


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
