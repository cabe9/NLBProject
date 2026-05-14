"""Experiment orchestration.

Responsibilities:

- load NWB data through :class:`nlb_tools.nwb_interface.NWBDataset`,
- build train/eval tensors via :mod:`nlb_tools.make_tensors`,
- run a reference (baseline) fit and a CV-selected fit for the requested
  :class:`nlb_project.model_registry.ModelSpec`,
- score both under :func:`nlb_tools.evaluation.evaluate`,
- write tracked artifacts (``metrics.csv``, ``ablation.csv``, ``summary.md``,
  ``run_metadata.json``) plus prediction HDF5s under ``cfg.output_dir``.

The per-model bookkeeping (which params are required, which axes are swept,
what kwargs the predict function takes) lives in :mod:`model_registry`; this
file contains no per-model branches.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from collections.abc import Iterator
from hashlib import sha256
from typing import Any

import numpy as np
from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_eval_target_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

from .config import ExperimentConfig
from .data_contract import resolve_data_path
from .io_utils import ensure_dir, write_metrics_csv, write_summary_md
from .model_registry import MODEL_REGISTRY, ModelSpec, get_spec
from .run_metadata import build_run_metadata

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -------- param assembly ----------------------------------------------------


def _rate_head_params(section: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Pull rate-readout params from a baseline/improvement section.

    Resolution order for each key:

    1. explicit value in ``section`` (so a sweep may override the readout
       without editing the top-level config),
    2. the top-level :class:`ExperimentConfig` fields
       (``cfg.output_head``, ``cfg.log_offset``).
    """
    return {
        "output_head": str(section.get("output_head", cfg.output_head)),
        "log_offset": float(section.get("log_offset", cfg.log_offset)),
    }


def _baseline_scalar_params(spec: ModelSpec, cfg: ExperimentConfig) -> dict[str, Any]:
    """Cast every ``spec.baseline_params`` key out of ``cfg.baseline``.

    Missing keys raise ``KeyError``; the registry enumerates exactly the
    keys each model requires, so an underspecified YAML fails fast.
    """
    out: dict[str, Any] = {}
    for name, caster in spec.baseline_params:
        if name not in cfg.baseline:
            raise KeyError(f"Missing required baseline key `{name}` for model `{spec.name}`")
        out[name] = caster(cfg.baseline[name])
    return out


def _rate_head_or_log_offset(
    spec: ModelSpec, cfg: ExperimentConfig, section: dict[str, Any]
) -> dict[str, Any]:
    """Return the appropriate rate-readout extras for ``spec``.

    Models that use a pluggable rate head get ``output_head`` + ``log_offset``;
    smoothing gets a bare ``log_offset`` (it uses its own Poisson head).
    Models declaring neither receive ``{}``.
    """
    if spec.uses_rate_head:
        return _rate_head_params(section, cfg)
    if spec.passes_log_offset:
        return {"log_offset": float(cfg.log_offset)}
    return {}


def build_reference_params(spec: ModelSpec, cfg: ExperimentConfig) -> dict[str, Any]:
    """Assemble the reference (baseline) params dict, exactly as serialized."""
    params = _baseline_scalar_params(spec, cfg)
    params.update(_rate_head_or_log_offset(spec, cfg, cfg.baseline))
    return params


def _apply_improvement_overrides(
    spec: ModelSpec, cfg: ExperimentConfig, base: dict[str, Any]
) -> dict[str, Any]:
    """Overlay ``cfg.improvement`` values for keys in ``spec.improvement_overrides``.

    Missing keys in improvement fall back to the value already present in
    ``base`` (which came from baseline), preserving the historical semantics.
    """
    out = dict(base)
    for name, caster in spec.improvement_overrides:
        if name in cfg.improvement:
            out[name] = caster(cfg.improvement[name])
    return out


def iter_cv_candidates(
    spec: ModelSpec, cfg: ExperimentConfig
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield ``(params_dict, label)`` pairs for every CV candidate."""
    axes: list[Any] = []
    grids: list[list[Any]] = []
    for axis in spec.sweep_axes:
        if axis.grid_key not in cfg.improvement:
            raise KeyError(
                f"Missing required improvement key `{axis.grid_key}` for model `{spec.name}`"
            )
        axes.append(axis)
        grids.append([axis.caster(v) for v in cfg.improvement[axis.grid_key]])
    for axis in spec.optional_sweep_axes:
        if axis.grid_key not in cfg.improvement:
            continue
        axes.append(axis)
        grids.append([axis.caster(v) for v in cfg.improvement[axis.grid_key]])

    base = _apply_improvement_overrides(spec, cfg, _baseline_scalar_params(spec, cfg))
    head_extras = _rate_head_or_log_offset(spec, cfg, cfg.improvement)

    for values in itertools.product(*grids):
        params = dict(base)
        label_parts: list[str] = []
        for axis, v in zip(axes, values, strict=True):
            params[axis.param_name] = v
            label_parts.append(f"{axis.param_name}={v}")
        params.update(head_extras)
        yield params, f"cv({','.join(label_parts)})"


# -------- CV bookkeeping ---------------------------------------------------


def _dataset_key(dataset_name: str, bin_size_ms: int) -> str:
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    return f"{dataset_name}{suf}"


def _split_key(dataset_name: str, bin_size_ms: int) -> str:
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    if "maze_" in dataset_name:
        return f"mc_maze_scaling{suf}_split"
    return f"{dataset_name}{suf}_split"


def _build_cv_masks(dataset: NWBDataset, split_name: str, n_folds: int, seed: int):
    all_idx = np.where(dataset.trial_info.split.to_numpy() == split_name)[0]
    rng = np.random.default_rng(seed)
    shuffled = all_idx.copy()
    rng.shuffle(shuffled)

    folds = []
    for fold_idx in range(n_folds):
        eval_idx = shuffled[fold_idx::n_folds]
        train_idx = np.setdiff1d(shuffled, eval_idx)
        train_mask = np.isin(np.arange(len(dataset.trial_info)), train_idx)
        eval_mask = np.isin(np.arange(len(dataset.trial_info)), eval_idx)
        folds.append((train_mask, eval_mask))
    return folds


# -------- evaluation --------------------------------------------------------


def _run_single_eval(
    dataset: NWBDataset,
    cfg: ExperimentConfig,
    spec: ModelSpec,
    train_split,
    eval_split,
    params: dict[str, Any],
    include_psth: bool,
    run_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logger.info("[%s] model_type=%s effective_params=%s", run_name, spec.name, params)
    train_dict = make_train_input_tensors(
        dataset, cfg.dataset_name, trial_split=train_split, save_file=False
    )
    eval_dict = make_eval_input_tensors(
        dataset, cfg.dataset_name, trial_split=eval_split, save_file=False
    )
    target_dict = make_eval_target_tensors(
        dataset,
        cfg.dataset_name,
        train_trial_split=train_split,
        eval_trial_split=eval_split,
        save_file=False,
        include_psth=include_psth,
    )

    extra_kwargs = spec.extra_predict_kwargs_fn(cfg)
    preds = spec.predict(
        train_dict["train_spikes_heldin"],
        train_dict["train_spikes_heldout"],
        eval_dict["eval_spikes_heldin"],
        **params,
        **extra_kwargs,
    )

    output_dict = {_dataset_key(cfg.dataset_name, cfg.bin_size_ms): preds}
    metrics = evaluate(target_dict, output_dict)[0][_split_key(cfg.dataset_name, cfg.bin_size_ms)]
    if not np.isfinite(metrics["co-bps"]):
        raise ValueError("co-bps is not finite; check preprocessing/model outputs")
    return output_dict, metrics


def _select_best_params(
    dataset: NWBDataset,
    cfg: ExperimentConfig,
    spec: ModelSpec,
) -> dict[str, Any]:
    """Generic CV grid search over every ``spec.sweep_axes`` axis."""
    cv_folds = int(cfg.improvement.get("cv_folds", spec.default_cv_folds))
    folds = _build_cv_masks(dataset, cfg.train_split, cv_folds, cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for params, label in iter_cv_candidates(spec, cfg):
        fold_scores: list[float] = []
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                spec,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=label,
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info("CV candidate %s -> mean co-bps %.4f", label, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None, f"No CV candidates for model `{spec.name}`"
    logger.info(
        "Selected params for %s: %s (cv mean co-bps %.4f)",
        spec.name,
        best_params,
        best_score,
    )
    return best_params


# -------- top-level driver --------------------------------------------------


def run_full_experiment(cfg: ExperimentConfig, *, config_path: str | None = None) -> dict[str, Any]:
    set_seeds(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    pred_dir = ensure_dir(out_dir / "predictions")

    spec = get_spec(cfg.model_type)

    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(dataset_path, cfg.data_prefix, skip_fields=cfg.skip_fields)
    dataset.resample(cfg.bin_size_ms)

    reference_params = build_reference_params(spec, cfg)
    selected_params = _select_best_params(dataset, cfg, spec)

    reference_output, reference_metrics = _run_single_eval(
        dataset,
        cfg,
        spec,
        cfg.train_split,
        cfg.eval_split,
        reference_params,
        include_psth=cfg.include_psth,
        run_name="reference",
    )
    reference_path = pred_dir / "baseline_predictions.h5"
    save_to_h5(reference_output, str(reference_path), overwrite=True)

    selected_output, selected_metrics = _run_single_eval(
        dataset,
        cfg,
        spec,
        cfg.train_split,
        cfg.eval_split,
        selected_params,
        include_psth=cfg.include_psth,
        run_name="selected",
    )
    selected_path = pred_dir / "improved_predictions.h5"
    save_to_h5(selected_output, str(selected_path), overwrite=True)

    reference_hash = sha256(reference_path.read_bytes()).hexdigest()
    selected_hash = sha256(selected_path.read_bytes()).hexdigest()
    logger.info(
        "Prediction artifact sha256 reference=%s selected=%s", reference_hash, selected_hash
    )
    params_differ = reference_params != selected_params
    if params_differ and reference_hash == selected_hash:
        logger.warning(
            "Reference and selected prediction files are byte-identical despite different params. "
            "This may indicate a parameter propagation bug."
        )
    elif not params_differ:
        logger.info("Selected params match reference params; identical artifacts are expected.")

    rows = [
        {
            "model": "baseline",
            "model_type": cfg.model_type,
            "co-bps": reference_metrics.get("co-bps"),
            "vel R2": reference_metrics.get("vel R2"),
            "psth R2": reference_metrics.get("psth R2"),
            "params": json.dumps(reference_params, sort_keys=True),
        },
        {
            "model": "improved",
            "model_type": cfg.model_type,
            "co-bps": selected_metrics.get("co-bps"),
            "vel R2": selected_metrics.get("vel R2"),
            "psth R2": selected_metrics.get("psth R2"),
            "params": json.dumps(selected_params, sort_keys=True),
        },
    ]
    write_metrics_csv(rows, out_dir / "ablation.csv")
    write_metrics_csv(rows, out_dir / "metrics.csv")
    write_summary_md(rows, out_dir / "summary.md")

    prediction_artifacts = {
        "baseline_predictions": {
            "path": str(reference_path),
            "sha256": reference_hash,
        },
        "improved_predictions": {
            "path": str(selected_path),
            "sha256": selected_hash,
        },
    }
    repro = build_run_metadata(
        cfg,
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=out_dir,
        baseline_metrics=reference_metrics,
        improved_metrics=selected_metrics,
        baseline_params=reference_params,
        improved_params=selected_params,
        prediction_artifacts=prediction_artifacts,
    )
    (out_dir / "run_metadata.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    return repro


__all__ = [
    "MODEL_REGISTRY",
    "build_reference_params",
    "iter_cv_candidates",
    "run_full_experiment",
    "set_seeds",
]
