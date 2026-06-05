"""Minimal LFADS train/val smoke test (no public-test, no sweep)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-h5",
        type=Path,
        default=None,
        help="LFADS HDF5 (default: data/lfads/mc_maze_20ms_val_smoke.h5 or full val file)",
    )
    parser.add_argument(
        "--lfads-torch-dir",
        type=Path,
        default=None,
        help="lfads-torch clone (default: LFADS_TORCH_DIR or external/lfads-torch)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory (default: results/lfads_smoke/<timestamp>)",
    )
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr-init",
        type=float,
        default=None,
        help="Override model.lr_init (default: smoke_single.yaml / nlb_mc_maze)",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Override trainer.gradient_clip_val (default: smoke_single.yaml)",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only verify imports and data")
    parser.add_argument(
        "--bin-size-ms",
        type=int,
        choices=(5, 20),
        default=20,
        help="Bin width label for defaults and manifest (default: 20)",
    )
    return parser.parse_args(argv)


def _ensure_ray_stub_for_single_run() -> None:
    """Avoid heavy ray[tune] + pandas/pyarrow pins for single-session smoke runs."""
    import types

    if "ray" in sys.modules:
        return
    tune_mod = types.ModuleType("ray.tune")
    tune_mod.get_trial_name = lambda: "smoke"  # type: ignore[attr-defined]
    ray_mod = types.ModuleType("ray")
    ray_mod.tune = tune_mod
    sys.modules["ray"] = ray_mod
    sys.modules["ray.tune"] = tune_mod


def _gpu_mem_mb() -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False, "allocated_mb": None, "reserved_mb": None}
        return {
            "cuda_available": True,
            "allocated_mb": float(torch.cuda.memory_allocated() / 2**20),
            "reserved_mb": float(torch.cuda.memory_reserved() / 2**20),
        }
    except Exception:
        return {"cuda_available": False, "allocated_mb": None, "reserved_mb": None}


def _resolve_data_h5(path: Path | None, bin_size_ms: int = 20) -> Path:
    root = _repo_root()
    if path is not None:
        return path.expanduser().resolve()
    smoke = root / "data" / "lfads" / f"mc_maze_{bin_size_ms}ms_val_smoke.h5"
    full = root / "data" / "lfads" / f"mc_maze_{bin_size_ms}ms_val.h5"
    if smoke.is_file():
        return smoke
    if full.is_file():
        logger.warning("Smoke subset missing; using full val HDF5 (still capped by max_epochs).")
        return full
    raise FileNotFoundError(
        "No LFADS HDF5 found. Run:\n"
        "  python scripts/prepare_lfads_mc_maze.py --write-smoke-subset"
    )


def _lfads_torch_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg.expanduser().resolve()
    env = os.environ.get("LFADS_TORCH_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "external" / "lfads-torch"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    sys.path.insert(0, str(_repo_root() / "scripts"))
    from lfads_data_utils import (
        analyze_lfads_metrics_csv,
        assert_finite_h5,
        best_checkpoint_in_dir,
        inspect_h5,
        lfads_training_overrides,
        model_dims_from_h5,
    )

    data_h5 = _resolve_data_h5(args.data_h5, args.bin_size_ms)
    lfads_dir = _lfads_torch_dir(args.lfads_torch_dir)
    if not (lfads_dir / "lfads_torch").is_dir():
        raise FileNotFoundError(f"lfads-torch not found at {lfads_dir}")

    import torch

    import lfads_torch  # noqa: F401
    import nlb_tools  # noqa: F401

    logger.info("torch %s | lfads_torch %s | nlb_tools ok", torch.__version__, lfads_dir)

    assert_finite_h5(data_h5)
    shapes = inspect_h5(data_h5)
    logger.info("Data shapes: %s", shapes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.output_dir or (_repo_root() / "results" / "lfads_smoke" / stamp)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_ms = args.bin_size_ms
    training_config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "lr_init": args.lr_init,
        "gradient_clip_val": args.gradient_clip_val,
    }
    manifest: dict = {
        "status": "started",
        "bin_size_ms": bin_ms,
        "bin_size_label": (
            f"{bin_ms} ms LFADS — label bin size on every score; "
            "5 ms aligns with STNDT-lite, 20 ms is lfads-torch reference"
        ),
        "data_h5": str(data_h5),
        "model_dims": model_dims_from_h5(data_h5),
        "training_config": training_config,
        "shapes": shapes,
        "lfads_torch_dir": str(lfads_dir),
        "gpu_before": _gpu_mem_mb(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.skip_train:
        manifest["status"] = "imports_and_data_only"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Skip-train mode; manifest at %s", out_dir / "manifest.json")
        return

    run_subdir = out_dir / "lfads_run"
    run_subdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), run_subdir / Path(__file__).name)

    sys.path.insert(0, str(lfads_dir))
    train_cwd = lfads_dir

    # run_model imports ray.tune at module load; smoke only needs single-session training.
    _ensure_ray_stub_for_single_run()

    from lfads_torch.run_model import run_model

    smoke_config_src = _repo_root() / "configs" / "lfads" / "smoke_single.yaml"
    smoke_config_dst = lfads_dir / "configs" / "smoke_single.yaml"
    shutil.copy2(smoke_config_src, smoke_config_dst)

    overrides = lfads_training_overrides(
        data_h5,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
        lr_init=args.lr_init,
        gradient_clip_val=args.gradient_clip_val,
    )

    logger.info("Training with cwd=%s for %s epochs", train_cwd, args.max_epochs)
    prev_cwd = Path.cwd()
    os.chdir(train_cwd)
    try:
        run_model(
            overrides=overrides,
            config_path="../configs/smoke_single.yaml",
            do_train=True,
            do_posterior_sample=False,
        )
    finally:
        os.chdir(prev_cwd)

    output_files = sorted(train_cwd.glob("lfads_output*.h5"))
    for out_h5 in output_files:
        dest = run_subdir / out_h5.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(out_h5), str(dest))
    output_files = sorted(run_subdir.glob("lfads_output*.h5"))

    ckpt_dir = train_cwd / "lightning_checkpoints"
    if ckpt_dir.is_dir():
        dest_ckpt = run_subdir / "lightning_checkpoints"
        if dest_ckpt.exists():
            shutil.rmtree(dest_ckpt)
        shutil.copytree(ckpt_dir, dest_ckpt)

    nan_report: dict[str, bool] = {}
    output_shapes: dict[str, dict] = {}
    for out_h5 in output_files:
        with __import__("h5py").File(out_h5, "r") as h5file:
            for key in ("train_output_params", "valid_output_params"):
                if key in h5file:
                    arr = np.asarray(h5file[key][()], dtype=np.float64)
                    nan_report[f"{out_h5.name}:{key}"] = bool(np.isfinite(arr).all())
                    output_shapes[f"{out_h5.name}:{key}"] = list(arr.shape)

    metrics_src = train_cwd / "csv_logs" / "metrics.csv"
    if not metrics_src.is_file():
        metrics_src = train_cwd / "csv_logs" / "version_0" / "metrics.csv"
    metrics_dest = run_subdir / "metrics.csv"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, metrics_dest)

    ckpt_dir_saved = run_subdir / "lightning_checkpoints"
    best_ckpt = best_checkpoint_in_dir(ckpt_dir_saved)
    metrics_summary = analyze_lfads_metrics_csv(metrics_dest)
    ckpt_saved = ckpt_dir_saved.is_dir()
    status = "completed"
    if metrics_summary.get("diverged"):
        status = "diverged"
    elif not ckpt_saved and not output_files:
        status = "completed_with_warnings"

    manifest.update(
        {
            "status": status,
            "output_files": [str(p) for p in output_files],
            "output_shapes": output_shapes,
            "finite_outputs": nan_report,
            "checkpoint_saved": ckpt_saved,
            "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
            "metrics_summary": metrics_summary,
            "gpu_after": _gpu_mem_mb(),
        }
    )
    if status == "completed_with_warnings":
        manifest["warning"] = (
            "No checkpoints or lfads_output HDF5; try smaller --batch-size or more smoke trials."
        )
    if status == "diverged":
        manifest["warning"] = (
            f"Training diverged at epoch {metrics_summary.get('first_nan_epoch')}; "
            f"last finite epoch {metrics_summary.get('last_finite_epoch')}."
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Smoke complete. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
