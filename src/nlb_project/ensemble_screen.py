"""Validation-only prediction averaging screens.

This module intentionally sits above the model layer: it reuses registered
``fit_predict_*`` functions, evaluates their predictions on train/val, and
optionally averages predictions from multiple recipes. It does not alter model
architecture or training internals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_eval_target_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

from .data_contract import resolve_data_path
from .io_utils import ensure_dir
from .model_registry import get_spec
from .pipeline import _dataset_key, _rate_head_or_log_offset, _split_key
from .public_test import PUBLIC_TEST_EVAL_DATA_URL, _metrics_for_split, sha256_file
from .run_metadata import collect_git_metadata, collect_runtime_metadata

logger = logging.getLogger(__name__)
_TORCH_MODEL_TYPES = {"ndt_lite", "ndt_factorized", "stndt_lite", "stndt_axial"}


@dataclass(frozen=True)
class EnsembleScreenConfig:
    """Validated config for a prediction-averaging validation screen."""

    model_type: str
    dataset_name: str
    data_path: str | None
    data_prefix: str
    bin_size_ms: int
    train_split: str
    eval_split: str
    include_psth: bool
    log_offset: float
    seed: int
    skip_fields: list[str]
    baseline: dict[str, Any]
    common_params: dict[str, Any]
    recipes: dict[str, dict[str, Any]]
    seeds: list[int]
    ensembles: dict[str, list[str]]
    gate: dict[str, Any]
    output_dir: str
    output_head: str = "log_link"


_REQUIRED_KEYS = {
    "model_type",
    "dataset_name",
    "data_path",
    "data_prefix",
    "bin_size_ms",
    "train_split",
    "eval_split",
    "include_psth",
    "log_offset",
    "seed",
    "skip_fields",
    "baseline",
    "common_params",
    "recipes",
    "seeds",
    "ensembles",
    "gate",
    "output_dir",
}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"output_head"}


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected mapping, got {value!r}")
    return dict(value)


def load_ensemble_screen_config(path: str | Path) -> EnsembleScreenConfig:
    """Load and validate an ensemble-screen YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping")

    missing = sorted(_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"Invalid config {path}: missing required keys {missing}")
    extras = sorted(set(raw) - _ALLOWED_KEYS)
    if extras:
        raise ValueError(f"Invalid config {path}: unknown keys {extras}")

    recipes = _require_mapping(raw["recipes"], label="recipes")
    if not recipes:
        raise ValueError(f"Invalid config {path}: recipes must not be empty")
    for recipe_name, recipe_params in recipes.items():
        _require_mapping(recipe_params, label=f"recipes.{recipe_name}")

    seeds_raw = raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"Invalid config {path}: seeds must be a non-empty list")
    seeds = [int(seed) for seed in seeds_raw]

    ensembles_raw = _require_mapping(raw["ensembles"], label="ensembles")
    ensembles: dict[str, list[str]] = {}
    for ensemble_name, recipe_names in ensembles_raw.items():
        if not isinstance(recipe_names, list) or len(recipe_names) < 2:
            raise ValueError(
                f"Invalid config {path}: ensembles.{ensemble_name} "
                "must list at least two recipe names"
            )
        names = [str(name) for name in recipe_names]
        unknown = sorted(set(names) - set(recipes))
        if unknown:
            raise ValueError(
                f"Invalid config {path}: ensembles.{ensemble_name} "
                f"references unknown recipes {unknown}"
            )
        ensembles[str(ensemble_name)] = names

    return EnsembleScreenConfig(
        model_type=str(raw["model_type"]),
        dataset_name=str(raw["dataset_name"]),
        data_path=None if raw["data_path"] is None else str(raw["data_path"]),
        data_prefix=str(raw["data_prefix"]),
        bin_size_ms=int(raw["bin_size_ms"]),
        train_split=str(raw["train_split"]),
        eval_split=str(raw["eval_split"]),
        include_psth=bool(raw["include_psth"]),
        log_offset=float(raw["log_offset"]),
        seed=int(raw["seed"]),
        skip_fields=[str(x) for x in raw["skip_fields"]],
        baseline=_require_mapping(raw["baseline"], label="baseline"),
        common_params=_require_mapping(raw["common_params"], label="common_params"),
        recipes={str(name): dict(params) for name, params in recipes.items()},
        seeds=seeds,
        ensembles=ensembles,
        gate=_require_mapping(raw["gate"], label="gate"),
        output_dir=str(raw["output_dir"]),
        output_head=str(raw.get("output_head", "log_link")),
    )


def _average_prediction_dicts(predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Average NLB prediction dictionaries by key."""
    if len(predictions) < 2:
        raise ValueError("At least two prediction dictionaries are required")
    keys = set(predictions[0])
    for pred in predictions[1:]:
        if set(pred) != keys:
            raise ValueError("Prediction dictionaries have mismatched keys")
    return {
        key: np.mean([pred[key] for pred in predictions], axis=0).astype(np.float32)
        for key in sorted(keys)
    }


def _score_output(
    target_dict: dict[str, Any],
    output_dict: dict[str, Any],
    *,
    dataset_name: str,
    bin_size_ms: int,
) -> dict[str, Any]:
    metrics = evaluate(target_dict, output_dict)[0][_split_key(dataset_name, bin_size_ms)]
    if not np.isfinite(metrics["co-bps"]):
        raise ValueError("co-bps is not finite; check prediction outputs")
    return metrics


def _aggregate_repeat_rows(
    rows: list[dict[str, Any]],
    *,
    gate_mean: float,
    min_repeats_above: int,
) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    repeat_rows = [row for row in rows if row["role"] in {"single_repeat", "ensemble_repeat"}]
    for (role, name), group in pd.DataFrame(repeat_rows).groupby(["role", "name"], sort=True):
        scores = group["co-bps"].astype(float).to_numpy()
        repeats_above = int(np.sum(scores > gate_mean))
        aggregate_rows.append(
            {
                "model": "aggregate",
                "model_type": "stndt_lite",
                "role": role.replace("_repeat", "_aggregate"),
                "name": name,
                "seed": "mean",
                "co-bps": float(np.mean(scores)),
                "vel R2": float(group["vel R2"].astype(float).mean()),
                "psth R2": None,
                "params": json.dumps(
                    {
                        "mean_co_bps": float(np.mean(scores)),
                        "std_co_bps": float(np.std(scores, ddof=0)),
                        "min_co_bps": float(np.min(scores)),
                        "max_co_bps": float(np.max(scores)),
                        "repeats": int(len(scores)),
                        "repeats_above_gate": repeats_above,
                        "gate_mean": gate_mean,
                        "min_repeats_above": min_repeats_above,
                        "passes_gate": bool(
                            role == "ensemble_repeat"
                            and np.mean(scores) > gate_mean
                            and repeats_above >= min_repeats_above
                        ),
                    },
                    sort_keys=True,
                ),
            }
        )
    return aggregate_rows


def _write_leaderboard(
    aggregate_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"ensemble_diversity_leaderboard_{stamp}.txt"
    ranked = sorted(aggregate_rows, key=lambda row: float(row["co-bps"]), reverse=True)
    lines = [
        "rank\tmean_co_bps\trole\tname\tseed_scores\tpasses_gate",
    ]
    repeats = pd.DataFrame(repeat_rows)
    for rank, row in enumerate(ranked, start=1):
        params = json.loads(row["params"])
        seed_scores = []
        if not repeats.empty:
            matched = repeats[(repeats["role"] == row["role"].replace("_aggregate", "_repeat"))]
            matched = matched[matched["name"] == row["name"]]
            seed_scores = [
                f"{int(seed)}:{score:.6f}"
                for seed, score in zip(matched["seed"], matched["co-bps"], strict=True)
            ]
        lines.append(
            f"{rank}\t{float(row['co-bps']):.6f}\t{row['role']}\t{row['name']}\t"
            f"{','.join(seed_scores)}\t{params['passes_gate']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote ensemble diversity leaderboard -> %s", path)
    return path


def run_ensemble_screen(
    cfg: EnsembleScreenConfig,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a validation-only recipe-diversity ensemble screen."""
    out_dir = ensure_dir(cfg.output_dir)
    pred_dir = ensure_dir(out_dir / "predictions")
    spec = get_spec(cfg.model_type)
    if cfg.model_type in _TORCH_MODEL_TYPES:
        import torch  # noqa: F401

        logger.info("Warm-imported torch before loading NLB data.")
    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(dataset_path, cfg.data_prefix, skip_fields=cfg.skip_fields)
    dataset.resample(cfg.bin_size_ms)

    train_dict = make_train_input_tensors(
        dataset, cfg.dataset_name, trial_split=cfg.train_split, save_file=False
    )
    eval_dict = make_eval_input_tensors(
        dataset, cfg.dataset_name, trial_split=cfg.eval_split, save_file=False
    )
    target_dict = make_eval_target_tensors(
        dataset,
        cfg.dataset_name,
        train_trial_split=cfg.train_split,
        eval_trial_split=cfg.eval_split,
        save_file=False,
        include_psth=cfg.include_psth,
    )

    dataset_key = _dataset_key(cfg.dataset_name, cfg.bin_size_ms)
    extra_kwargs = spec.extra_predict_kwargs_fn(cfg)
    rows: list[dict[str, Any]] = []
    member_predictions: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for seed in cfg.seeds:
        member_predictions[seed] = {}
        for recipe_name, recipe_params in cfg.recipes.items():
            params = dict(cfg.common_params)
            params.update(recipe_params)
            params["seed"] = seed
            params.update(_rate_head_or_log_offset(spec, cfg, params))
            logger.info("[%s seed=%s] effective_params=%s", recipe_name, seed, params)
            prediction = spec.predict(
                train_dict["train_spikes_heldin"],
                train_dict["train_spikes_heldout"],
                eval_dict["eval_spikes_heldin"],
                **params,
                **extra_kwargs,
            )
            member_predictions[seed][recipe_name] = prediction
            metrics = _score_output(
                target_dict,
                {dataset_key: prediction},
                dataset_name=cfg.dataset_name,
                bin_size_ms=cfg.bin_size_ms,
            )
            rows.append(
                {
                    "model": "candidate",
                    "model_type": cfg.model_type,
                    "role": "single_repeat",
                    "name": recipe_name,
                    "seed": seed,
                    "co-bps": metrics.get("co-bps"),
                    "vel R2": metrics.get("vel R2"),
                    "psth R2": metrics.get("psth R2"),
                    "params": json.dumps(params, sort_keys=True),
                }
            )
            logger.info(
                "[%s seed=%s] validation co-bps %.6f",
                recipe_name,
                seed,
                metrics["co-bps"],
            )

        for ensemble_name, recipe_names in cfg.ensembles.items():
            mixed_prediction = _average_prediction_dicts(
                [member_predictions[seed][recipe_name] for recipe_name in recipe_names]
            )
            metrics = _score_output(
                target_dict,
                {dataset_key: mixed_prediction},
                dataset_name=cfg.dataset_name,
                bin_size_ms=cfg.bin_size_ms,
            )
            rows.append(
                {
                    "model": "candidate",
                    "model_type": cfg.model_type,
                    "role": "ensemble_repeat",
                    "name": ensemble_name,
                    "seed": seed,
                    "co-bps": metrics.get("co-bps"),
                    "vel R2": metrics.get("vel R2"),
                    "psth R2": metrics.get("psth R2"),
                    "params": json.dumps(
                        {"recipes": recipe_names, "seed": seed, "weights": "equal"},
                        sort_keys=True,
                    ),
                }
            )
            logger.info(
                "[%s seed=%s] mixed validation co-bps %.6f",
                ensemble_name,
                seed,
                metrics["co-bps"],
            )

    gate_mean = float(cfg.gate.get("mean_co_bps", 0.3704))
    min_repeats_above = int(cfg.gate.get("min_repeats_above", 2))
    aggregate_rows = _aggregate_repeat_rows(
        rows,
        gate_mean=gate_mean,
        min_repeats_above=min_repeats_above,
    )

    ensemble_aggregates = [
        row for row in aggregate_rows if row["role"] == "ensemble_aggregate"
    ]
    single_aggregates = [row for row in aggregate_rows if row["role"] == "single_aggregate"]
    best_ensemble = max(ensemble_aggregates, key=lambda row: float(row["co-bps"]))
    anchor_row = next(
        (row for row in single_aggregates if row["name"] == "anchor"),
        max(single_aggregates, key=lambda row: float(row["co-bps"])),
    )

    metrics_rows = [
        {
            "model": "baseline",
            "model_type": cfg.model_type,
            "role": "single_aggregate",
            "name": anchor_row["name"],
            "seed": "mean",
            "co-bps": anchor_row["co-bps"],
            "vel R2": anchor_row["vel R2"],
            "psth R2": anchor_row["psth R2"],
            "params": anchor_row["params"],
        },
        {
            "model": "improved",
            "model_type": cfg.model_type,
            "role": "ensemble_aggregate",
            "name": best_ensemble["name"],
            "seed": "mean",
            "co-bps": best_ensemble["co-bps"],
            "vel R2": best_ensemble["vel R2"],
            "psth R2": best_ensemble["psth R2"],
            "params": best_ensemble["params"],
        },
        *aggregate_rows,
        *rows,
    ]
    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    leaderboard_path = _write_leaderboard(aggregate_rows, rows, out_dir)

    # Save only the compact headline prediction artifacts for local provenance.
    anchor_seed = cfg.seeds[0]
    baseline_prediction = member_predictions[anchor_seed][str(anchor_row["name"])]
    best_prediction = _average_prediction_dicts(
        [
            member_predictions[anchor_seed][recipe_name]
            for recipe_name in cfg.ensembles[str(best_ensemble["name"])]
        ]
    )
    baseline_path = pred_dir / "baseline_predictions.h5"
    improved_path = pred_dir / "improved_predictions.h5"
    save_to_h5({dataset_key: baseline_prediction}, str(baseline_path), overwrite=True)
    save_to_h5({dataset_key: best_prediction}, str(improved_path), overwrite=True)

    summary = {
        "config_path": None if config_path is None else str(config_path),
        "output_dir": str(out_dir),
        "metrics_path": str(metrics_path),
        "leaderboard_path": str(leaderboard_path),
        "gate": cfg.gate,
        "baseline": anchor_row,
        "best_ensemble": best_ensemble,
        "passes_gate": json.loads(best_ensemble["params"])["passes_gate"],
    }
    (out_dir / "summary.md").write_text(_format_summary(summary, aggregate_rows), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "metadata_schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "config_path": None if config_path is None else str(config_path),
                "config": cfg.__dict__,
                "runtime": collect_runtime_metadata(),
                "git": collect_git_metadata(),
                "artifacts": {
                    "output_dir": str(out_dir),
                    "predictions": {
                        "baseline_predictions": {
                            "path": str(baseline_path),
                            "sha256": sha256(baseline_path.read_bytes()).hexdigest(),
                        },
                        "improved_predictions": {
                            "path": str(improved_path),
                            "sha256": sha256(improved_path.read_bytes()).hexdigest(),
                        },
                    },
                },
                "baseline_metrics": {
                    "co-bps": anchor_row["co-bps"],
                    "vel R2": anchor_row["vel R2"],
                    "psth R2": anchor_row["psth R2"],
                },
                "improved_metrics": {
                    "co-bps": best_ensemble["co-bps"],
                    "vel R2": best_ensemble["vel R2"],
                    "psth R2": best_ensemble["psth R2"],
                },
                "baseline_params": json.loads(anchor_row["params"]),
                "improved_params": json.loads(best_ensemble["params"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def run_ensemble_public_test(
    cfg: EnsembleScreenConfig,
    *,
    eval_data_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    final_train_trial_split: list[str] | None = None,
    ensemble_name: str | None = None,
) -> dict[str, Any]:
    """Fit one mixed ensemble on train+val and score it once on public test."""
    final_train_trial_split = final_train_trial_split or ["train", "val"]
    eval_data_path = Path(eval_data_path)
    if not eval_data_path.exists():
        raise FileNotFoundError(
            f"Public test eval data not found at {eval_data_path}. "
            "Run `nlb-get-public-eval-data` first."
        )
    if len(cfg.seeds) != 1:
        raise ValueError("Public ensemble config must contain exactly one seed")

    selected_ensemble_name = ensemble_name or next(iter(cfg.ensembles))
    if selected_ensemble_name not in cfg.ensembles:
        raise ValueError(f"Unknown ensemble `{selected_ensemble_name}`")

    out_dir = ensure_dir(output_dir or cfg.output_dir)
    pred_dir = ensure_dir(out_dir / "predictions")
    spec = get_spec(cfg.model_type)
    if cfg.model_type in _TORCH_MODEL_TYPES:
        import torch  # noqa: F401

        logger.info("Warm-imported torch before loading NLB data.")

    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(dataset_path, cfg.data_prefix, skip_fields=cfg.skip_fields)
    dataset.resample(cfg.bin_size_ms)
    train_dict = make_train_input_tensors(
        dataset, cfg.dataset_name, trial_split=final_train_trial_split, save_file=False
    )
    eval_dict = make_eval_input_tensors(
        dataset, cfg.dataset_name, trial_split="test", save_file=False
    )

    seed = int(cfg.seeds[0])
    recipe_names = cfg.ensembles[selected_ensemble_name]
    predictions: list[dict[str, np.ndarray]] = []
    recipe_params_by_name: dict[str, dict[str, Any]] = {}
    extra_kwargs = spec.extra_predict_kwargs_fn(cfg)
    for recipe_name in recipe_names:
        params = dict(cfg.common_params)
        params.update(cfg.recipes[recipe_name])
        params["seed"] = seed
        params.update(_rate_head_or_log_offset(spec, cfg, params))
        recipe_params_by_name[recipe_name] = params
        logger.info(
            "[%s public-test ensemble=%s] effective_params=%s",
            recipe_name,
            selected_ensemble_name,
            params,
        )
        predictions.append(
            spec.predict(
                train_dict["train_spikes_heldin"],
                train_dict["train_spikes_heldout"],
                eval_dict["eval_spikes_heldin"],
                **params,
                **extra_kwargs,
            )
        )

    mixed_prediction = _average_prediction_dicts(predictions)
    dataset_key = _dataset_key(cfg.dataset_name, cfg.bin_size_ms)
    output_dict = {dataset_key: mixed_prediction}
    metrics = _metrics_for_split(
        eval_data_path,
        output_dict,
        dataset_name=cfg.dataset_name,
        bin_size_ms=cfg.bin_size_ms,
    )

    prediction_path = pred_dir / "selected_public_test_predictions.h5"
    save_to_h5(output_dict, str(prediction_path), overwrite=True)

    split_key = _split_key(cfg.dataset_name, cfg.bin_size_ms)
    final_train_label = "+".join(final_train_trial_split)
    params_payload = {
        "ensemble": selected_ensemble_name,
        "recipes": recipe_names,
        "seed": seed,
        "weights": "equal",
        "recipe_params": recipe_params_by_name,
    }
    rows = [
        {
            "model": "selected",
            "model_type": cfg.model_type,
            "split": split_key,
            "train_split": final_train_label,
            "model_selection_train_split": cfg.train_split,
            "model_selection_eval_split": cfg.eval_split,
            "eval_split": "test",
            "co-bps": metrics.get("co-bps"),
            "vel R2": metrics.get("vel R2"),
            "psth R2": metrics.get("psth R2"),
            "fp-bps": metrics.get("fp-bps"),
            "params": json.dumps(params_payload, sort_keys=True),
        }
    ]
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    _write_ensemble_public_summary(rows, out_dir / "summary.md")

    metadata = {
        "metadata_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "config_path": None if config_path is None else str(config_path),
        "config": cfg.__dict__,
        "runtime": collect_runtime_metadata(),
        "git": collect_git_metadata(),
        "data": {
            "dataset_name": cfg.dataset_name,
            "resolved_data_path": str(dataset_path),
            "data_prefix": cfg.data_prefix,
            "bin_size_ms": cfg.bin_size_ms,
            "train_split": final_train_label,
            "model_selection_train_split": cfg.train_split,
            "model_selection_eval_split": cfg.eval_split,
            "eval_split": "test",
        },
        "artifacts": {
            "output_dir": str(out_dir),
            "predictions": {
                "selected_public_test_predictions": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                }
            },
        },
        "public_test_eval_data": {
            "path": str(eval_data_path),
            "sha256": sha256_file(eval_data_path),
            "source_url": PUBLIC_TEST_EVAL_DATA_URL,
        },
        "selected_metrics": metrics,
        "selected_params": params_payload,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "selected_metrics": metrics,
        "selected_params": params_payload,
        "output_dir": out_dir,
        "prediction_path": prediction_path,
    }


def _write_ensemble_public_summary(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    lines = [
        "# NLB Ensemble Public Test Summary",
        "",
        "Evaluated locally against the public NLB test target HDF5.",
        "",
        "| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {co_bps} | {vel_r2} | {psth_r2} | {fp_bps} | {params} |".format(
                model=row["model"],
                split=row["split"],
                co_bps=fmt(row.get("co-bps")),
                vel_r2=fmt(row.get("vel R2")),
                psth_r2=fmt(row.get("psth R2")),
                fp_bps=fmt(row.get("fp-bps")),
                params=row["params"],
            )
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def _format_summary(summary: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> str:
    ranked = sorted(aggregate_rows, key=lambda row: float(row["co-bps"]), reverse=True)
    lines = [
        "# STNDT-lite Ensemble Diversity Screen",
        "",
        f"Config: `{summary['config_path']}`",
        f"Gate: mean co-bps > `{summary['gate'].get('mean_co_bps', 0.3704)}` "
        f"and at least `{summary['gate'].get('min_repeats_above', 2)}` repeats above gate.",
        "",
        "| rank | role | name | mean co-bps | passes gate |",
        "|---:|---|---|---:|---|",
    ]
    for rank, row in enumerate(ranked, start=1):
        params = json.loads(row["params"])
        lines.append(
            f"| {rank} | {row['role']} | {row['name']} | "
            f"{float(row['co-bps']):.6f} | {params['passes_gate']} |"
        )
    lines.extend(
        [
            "",
            f"Best ensemble: `{summary['best_ensemble']['name']}` "
            f"at `{float(summary['best_ensemble']['co-bps']):.6f}` co-bps.",
            f"Passes gate: `{summary['passes_gate']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "EnsembleScreenConfig",
    "load_ensemble_screen_config",
    "run_ensemble_public_test",
    "run_ensemble_screen",
    "_aggregate_repeat_rows",
    "_average_prediction_dicts",
]
