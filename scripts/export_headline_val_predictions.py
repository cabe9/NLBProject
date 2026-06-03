"""Fit headline ensemble recipes on train and export validation predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

from nlb_project.data_contract import resolve_data_path
from nlb_project.ensemble_screen import (
    EnsembleScreenConfig,
    _average_prediction_dicts,
    load_ensemble_screen_config,
)
from nlb_project.io_utils import ensure_dir
from nlb_project.model_registry import get_spec
from nlb_project.pipeline import _dataset_key, _rate_head_or_log_offset

logger = logging.getLogger(__name__)


def export_val_predictions(cfg: EnsembleScreenConfig) -> dict[str, Path]:
    """Train each recipe on train split and save validation predictions to HDF5."""
    out_dir = ensure_dir(cfg.output_dir)
    pred_dir = ensure_dir(out_dir / "predictions")
    spec = get_spec(cfg.model_type)
    if cfg.model_type in {"ndt_lite", "ndt_factorized", "stndt_lite", "stndt_axial"}:
        import torch  # noqa: F401

    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(dataset_path, cfg.data_prefix, skip_fields=cfg.skip_fields)
    dataset.resample(cfg.bin_size_ms)

    train_dict = make_train_input_tensors(
        dataset, cfg.dataset_name, trial_split=cfg.train_split, save_file=False
    )
    eval_dict = make_eval_input_tensors(
        dataset, cfg.dataset_name, trial_split=cfg.eval_split, save_file=False
    )
    dataset_key = _dataset_key(cfg.dataset_name, cfg.bin_size_ms)
    extra_kwargs = spec.extra_predict_kwargs_fn(cfg)
    seed = int(cfg.seeds[0])

    saved: dict[str, Path] = {}
    member_predictions: dict[str, dict[str, object]] = {}

    for recipe_name, recipe_params in cfg.recipes.items():
        params = dict(cfg.common_params)
        params.update(recipe_params)
        params["seed"] = seed
        params.update(_rate_head_or_log_offset(spec, cfg, params))
        logger.info("[%s] effective_params=%s", recipe_name, params)
        prediction = spec.predict(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            **params,
            **extra_kwargs,
        )
        member_predictions[recipe_name] = prediction
        path = pred_dir / f"{recipe_name}_seed{seed}_val_predictions.h5"
        save_to_h5({dataset_key: prediction}, str(path), overwrite=True)
        saved[recipe_name] = path
        logger.info("Wrote %s", path)

    for ensemble_name, recipe_names in cfg.ensembles.items():
        mixed = _average_prediction_dicts(
            [member_predictions[name] for name in recipe_names]  # type: ignore[list-item]
        )
        path = pred_dir / f"{ensemble_name}_seed{seed}_val_predictions.h5"
        save_to_h5({dataset_key: mixed}, str(path), overwrite=True)
        saved[ensemble_name] = path
        logger.info("Wrote %s", path)

    manifest = {
        "dataset_name": cfg.dataset_name,
        "bin_size_ms": cfg.bin_size_ms,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
        "seed": seed,
        "predictions": {name: str(path) for name, path in saved.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest -> %s", manifest_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export headline val predictions.")
    parser.add_argument(
        "--config",
        default="configs/diagnostics/mc_maze_headline_val_predictions.yaml",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = load_ensemble_screen_config(args.config)
    export_val_predictions(cfg)


if __name__ == "__main__":
    main()
