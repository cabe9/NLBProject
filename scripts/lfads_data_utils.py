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


def model_dims_from_h5(path: Path) -> dict[str, int]:
    """Infer LFADS model/datamodule dimensions from an MC_Maze HDF5."""
    with h5py.File(path, "r") as h5file:
        encod = h5file["train_encod_data"]
        recon = h5file["train_recon_data"]
        return {
            "encod_data_dim": int(encod.shape[-1]),
            "encod_seq_len": int(encod.shape[1]),
            "recon_seq_len": int(recon.shape[1]),
            "readout_out_features": int(recon.shape[-1]),
        }


def lfads_h5_has_psth(path: Path) -> bool:
    with h5py.File(path, "r") as h5file:
        return "psth" in h5file


def lfads_training_overrides(
    data_h5: Path,
    *,
    batch_size: int,
    max_epochs: int,
    seed: int = 0,
    lr_init: float | None = None,
    gradient_clip_val: float | None = None,
) -> dict[str, object]:
    """Hydra override dict for smoke_single.yaml from a prepared HDF5."""
    dims = model_dims_from_h5(data_h5)
    pattern = str(data_h5.resolve()).replace("\\", "/")
    overrides: dict[str, object] = {
        "datamodule.datafile_pattern": pattern,
        "datamodule.batch_size": batch_size,
        "trainer.max_epochs": max_epochs,
        "seed": seed,
        "model.encod_data_dim": dims["encod_data_dim"],
        "model.encod_seq_len": dims["encod_seq_len"],
        "model.recon_seq_len": dims["recon_seq_len"],
        "model.readout.modules.0.out_features": dims["readout_out_features"],
    }
    if lr_init is not None:
        overrides["model.lr_init"] = lr_init
    if gradient_clip_val is not None:
        overrides["trainer.gradient_clip_val"] = gradient_clip_val
    if not lfads_h5_has_psth(data_h5):
        overrides["datamodule.attr_keys"] = []
    return overrides


def best_checkpoint_in_dir(ckpt_dir: Path) -> Path | None:
    """Return the best (non-last) checkpoint saved by ModelCheckpoint, if any."""
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    non_last = [p for p in candidates if p.name != "last.ckpt"]
    return non_last[-1] if non_last else None


_DIVERGENCE_METRIC_KEYS = ("valid/recon_smth", "valid/loss", "valid/recon")
_PSTH_METRIC_KEYS = ("valid/r2", "train/r2")


def _is_validation_metrics_row(row: dict[str, str]) -> bool:
    """True for csv_logs validation rows (train rows leave valid/* empty)."""
    return any(row.get(key) not in (None, "") for key in _DIVERGENCE_METRIC_KEYS)


def _raw_metric_non_finite(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        return not np.isfinite(float(value))
    except ValueError:
        return True


def analyze_lfads_metrics_csv(
    path: Path,
    *,
    psth_available: bool | None = None,
) -> dict[str, Any]:
    """Summarize LFADS csv_logs metrics for divergence and best valid/recon_smth."""
    import csv

    if not path.is_file():
        return {"metrics_csv": str(path), "available": False}

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    def _float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    best_smth: float | None = None
    best_epoch: int | None = None
    last_finite_epoch: int | None = None
    first_nan_epoch: int | None = None
    nan_columns: list[str] = []
    validation_epochs_seen: list[int] = []

    for row in rows:
        if not _is_validation_metrics_row(row):
            continue
        epoch_raw = row.get("cur_epoch")
        if epoch_raw in (None, ""):
            epoch_raw = row.get("epoch")
        if epoch_raw in (None, ""):
            continue
        epoch = int(float(epoch_raw))
        validation_epochs_seen.append(epoch)

        core_non_finite = [
            key for key in _DIVERGENCE_METRIC_KEYS if _raw_metric_non_finite(row.get(key))
        ]
        if core_non_finite:
            if first_nan_epoch is None:
                first_nan_epoch = epoch
            nan_columns.extend(core_non_finite)
            continue

        smth = _float(row.get("valid/recon_smth"))
        loss = _float(row.get("valid/loss"))
        if smth is not None or loss is not None:
            last_finite_epoch = epoch
            if smth is not None and (best_smth is None or smth < best_smth):
                best_smth = smth
                best_epoch = epoch

    if psth_available is False:
        psth_metrics_status = "unavailable_no_psth_in_hdf5"
    elif psth_available is True:
        psth_metrics_status = "available"
    else:
        psth_metrics_status = "unknown"

    diverged = first_nan_epoch is not None
    return {
        "metrics_csv": str(path),
        "available": True,
        "best_valid_recon_smth": best_smth,
        "best_epoch": best_epoch,
        "last_finite_epoch": last_finite_epoch,
        "first_nan_epoch": first_nan_epoch,
        "diverged": diverged,
        "nan_columns": sorted(set(nan_columns)),
        "completed_epochs": last_finite_epoch,
        "validation_epochs_logged": len(validation_epochs_seen),
        "psth_metrics_status": psth_metrics_status,
    }


def assert_finite_h5(path: Path, keys: tuple[str, ...] = LFADS_H5_KEYS) -> None:
    with h5py.File(path, "r") as h5file:
        for key in keys:
            if key not in h5file:
                raise KeyError(f"Missing {key} in {path}")
            arr = np.asarray(h5file[key][()], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in {path}:{key}")
