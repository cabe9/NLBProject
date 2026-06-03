"""Shared helpers for LFADS MC_Maze HDF5 prep (no STNDT-lite imports)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

LFADS_H5_KEYS = (
    "train_encod_data",
    "train_recon_data",
    "valid_encod_data",
    "valid_recon_data",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_lfads_torch_dir() -> Path:
    env = __import__("os").environ.get("LFADS_TORCH_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "external" / "lfads-torch"


def reference_mc_maze_h5(lfads_torch_dir: Path, bin_size_ms: int) -> Path:
    if bin_size_ms == 20:
        return lfads_torch_dir / "datasets" / "mc_maze-20ms-val.h5"
    raise FileNotFoundError(
        f"No bundled lfads-torch reference HDF5 for bin_size_ms={bin_size_ms}. "
        "Use --source nwb or copy a prepared file manually."
    )


def stack_recon(
    heldin: np.ndarray,
    heldout: np.ndarray,
    heldin_forward: np.ndarray,
    heldout_forward: np.ndarray,
) -> np.ndarray:
    """Observed + forward-prediction blocks along time (NLB / AutoLFADS layout)."""
    main = np.concatenate([heldin, heldout], axis=-1)
    forward = np.concatenate([heldin_forward, heldout_forward], axis=-1)
    return np.concatenate([main, forward], axis=1).astype(np.float32)


def encod_from_heldin(heldin: np.ndarray) -> np.ndarray:
    return np.asarray(heldin, dtype=np.float32)


def copy_h5(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_manifest(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_h5(path: Path) -> dict[str, Any]:
    shapes: dict[str, list[int]] = {}
    with h5py.File(path, "r") as h5file:
        for key in sorted(h5file.keys()):
            ds = h5file[key]
            if hasattr(ds, "shape"):
                shapes[key] = list(ds.shape)
    return shapes


def subset_h5(
    src: Path,
    dst: Path,
    *,
    max_train_trials: int,
    max_valid_trials: int,
) -> dict[str, list[int]]:
    """Write a smaller HDF5 for smoke tests by slicing trial axis."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, list[int]] = {}
    with h5py.File(src, "r") as src_h5, h5py.File(dst, "w") as dst_h5:
        for key, ds in src_h5.items():
            data = ds[()]
            if key.startswith("train_") and data.ndim >= 1:
                n = min(max_train_trials, data.shape[0])
                data = data[:n]
            elif key.startswith("valid_") and data.ndim >= 1:
                n = min(max_valid_trials, data.shape[0])
                data = data[:n]
            elif key == "psth" and "train_encod_data" in src_h5:
                # psth is condition-averaged; keep as-is
                pass
            dst_h5.create_dataset(key, data=data, dtype=ds.dtype)
            if hasattr(data, "shape"):
                shapes[key] = list(data.shape)
    return shapes


def assert_finite_h5(path: Path, keys: tuple[str, ...] = LFADS_H5_KEYS) -> None:
    with h5py.File(path, "r") as h5file:
        for key in keys:
            if key not in h5file:
                raise KeyError(f"Missing {key} in {path}")
            arr = np.asarray(h5file[key][()], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in {path}:{key}")
