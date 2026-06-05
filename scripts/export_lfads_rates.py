"""Export LFADS rate predictions from a saved checkpoint (no training)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_ray_stub() -> None:
    import types

    if "ray" in sys.modules:
        return
    tune_mod = types.ModuleType("ray.tune")
    tune_mod.get_trial_name = lambda: "smoke"  # type: ignore[attr-defined]
    ray_mod = types.ModuleType("ray")
    ray_mod.tune = tune_mod
    sys.modules["ray"] = ray_mod
    sys.modules["ray.tune"] = tune_mod


def _lfads_torch_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg.expanduser().resolve()
    env = os.environ.get("LFADS_TORCH_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "external" / "lfads-torch"


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    ckpt_dir = run_dir / "lfads_run" / "lightning_checkpoints"
    if not ckpt_dir.is_dir():
        ckpt_dir = run_dir / "lightning_checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No lightning_checkpoints under {run_dir}")
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
    # Prefer step checkpoint over last.ckpt when both exist
    non_last = [p for p in candidates if p.name != "last.ckpt"]
    return non_last[-1] if non_last else candidates[-1]


def _resolve_data_h5(run_dir: Path, data_h5: Path | None) -> Path:
    if data_h5 is not None:
        return data_h5.expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("data_h5"):
            return Path(manifest["data_h5"]).resolve()
    smoke = _repo_root() / "data" / "lfads" / "mc_maze_20ms_val_smoke.h5"
    if smoke.is_file():
        return smoke
    raise FileNotFoundError("Pass --data-h5 or ensure run manifest.json has data_h5")


def export_rates(
    *,
    run_dir: Path,
    checkpoint: Path | None,
    data_h5: Path | None,
    lfads_torch_dir: Path | None,
    num_samples: int,
    output_subdir: str = "lfads_outputs",
) -> dict:
    run_dir = run_dir.expanduser().resolve()
    data_h5 = _resolve_data_h5(run_dir, data_h5)
    ckpt_path = _resolve_checkpoint(run_dir, checkpoint)
    lfads_dir = _lfads_torch_dir(lfads_torch_dir)

    sys.path.insert(0, str(_repo_root() / "scripts"))
    sys.path.insert(0, str(lfads_dir))
    from lfads_data_utils import inspect_h5, lfads_training_overrides

    out_dir = run_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_ray_stub()
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from lfads_torch.post_run.analysis import run_posterior_sampling
    from lfads_torch.run_model import flatten

    smoke_config_src = _repo_root() / "configs" / "lfads" / "smoke_single.yaml"
    smoke_config_dst = lfads_dir / "configs" / "smoke_single.yaml"
    if smoke_config_src.is_file():
        import shutil

        shutil.copy2(smoke_config_src, smoke_config_dst)

    overrides = lfads_training_overrides(
        data_h5,
        batch_size=4,
        max_epochs=1,
        seed=0,
    )
    override_list = [f"{k}={v}" for k, v in flatten(overrides).items()]

    prev_cwd = Path.cwd()
    os.chdir(lfads_dir)
    try:
        config_dir = str((lfads_dir / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            config = compose(config_name="smoke_single.yaml", overrides=override_list)
        datamodule = instantiate(config.datamodule, _convert_="all")
        model = instantiate(config.model)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        output_stem = out_dir / "lfads_output.h5"
        run_posterior_sampling(
            model,
            datamodule,
            filename=str(output_stem.resolve()),
            num_samples=int(num_samples),
        )
    finally:
        os.chdir(prev_cwd)

    # Posterior sampling writes lfads_output_<session>.h5 next to filename stem cwd = lfads_dir
    # Re-home outputs into run_dir/lfads_outputs/
    generated = sorted(Path(lfads_dir).glob("lfads_output_*.h5"), key=lambda p: p.stat().st_mtime)
    if not generated:
        generated = sorted(out_dir.glob("lfads_output_*.h5"))
    if not generated:
        raise FileNotFoundError(
            "Posterior sampling finished but no lfads_output_*.h5 found in "
            f"{lfads_dir} or {out_dir}"
        )
    session_h5 = generated[-1]
    if session_h5.parent != out_dir:
        dest = out_dir / session_h5.name
        if dest.exists():
            dest.unlink()
        session_h5.replace(dest)
        session_h5 = dest

    shapes = inspect_h5(session_h5)
    import numpy as np

    finite = {}
    for key in ("train_output_params", "valid_output_params"):
        if key in shapes:
            with __import__("h5py").File(session_h5, "r") as h5file:
                arr = np.asarray(h5file[key][()], dtype=np.float64)
                finite[key] = bool(np.isfinite(arr).all())

    bin_ms = 20
    root_manifest = run_dir / "manifest.json"
    if root_manifest.is_file():
        bin_ms = int(json.loads(root_manifest.read_text(encoding="utf-8")).get("bin_size_ms", 20))
    manifest = {
        "status": "exported",
        "bin_size_ms": bin_ms,
        "bin_size_label": f"{bin_ms} ms LFADS — label bin size on every score",
        "checkpoint": str(ckpt_path),
        "data_h5": str(data_h5),
        "lfads_output_h5": str(session_h5),
        "num_posterior_samples": num_samples,
        "shapes": shapes,
        "finite_output_params": finite,
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if root_manifest.is_file():
        root = json.loads(root_manifest.read_text(encoding="utf-8"))
        root["lfads_output_h5"] = str(session_h5)
        root["export_manifest"] = str(manifest_path)
        root_manifest.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")

    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Smoke run directory (e.g. results/lfads_smoke/20260603T074245Z)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-h5", type=Path, default=None)
    parser.add_argument("--lfads-torch-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=2, help="Posterior samples to average")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    manifest = export_rates(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        data_h5=args.data_h5,
        lfads_torch_dir=args.lfads_torch_dir,
        num_samples=args.num_samples,
    )
    logger.info("Exported %s", manifest["lfads_output_h5"])
    for key, shape in manifest["shapes"].items():
        if "output_params" in key:
            logger.info("  %s %s", key, shape)


if __name__ == "__main__":
    main()
