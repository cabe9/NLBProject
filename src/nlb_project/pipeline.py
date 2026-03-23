from __future__ import annotations

import itertools
import json
import logging
import random
from hashlib import sha256
from pathlib import Path
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
from .models import (
    predict_lagged_pca_latent_regression,
    predict_lagged_ridge_direct,
    predict_pca_latent_regression,
    predict_ridge_direct,
)
from .smoothing import SmoothingParams, predict_rates

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _dataset_key(dataset_name: str, bin_size_ms: int) -> str:
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    return f"{dataset_name}{suf}"


def _split_key(dataset_name: str, bin_size_ms: int) -> str:
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    if "maze_" in dataset_name:
        return f"mc_maze_scaling{suf}_split"
    return f"{dataset_name}{suf}_split"


def _run_single_eval(
    dataset: NWBDataset,
    cfg: ExperimentConfig,
    train_split,
    eval_split,
    params: dict[str, Any],
    include_psth: bool,
    run_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    logger.info("[%s] model_type=%s effective_params=%s", run_name, cfg.model_type, params)
    train_dict = make_train_input_tensors(
        dataset,
        cfg.dataset_name,
        trial_split=train_split,
        save_file=False,
    )
    eval_dict = make_eval_input_tensors(
        dataset,
        cfg.dataset_name,
        trial_split=eval_split,
        save_file=False,
    )
    target_dict = make_eval_target_tensors(
        dataset,
        cfg.dataset_name,
        train_trial_split=train_split,
        eval_trial_split=eval_split,
        save_file=False,
        include_psth=include_psth,
    )

    if cfg.model_type == "smoothing":
        smooth_params = SmoothingParams(
            kern_sd_ms=params["kern_sd_ms"],
            alpha=params["alpha"],
            log_offset=params["log_offset"],
        )
        preds = predict_rates(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            smooth_params,
            cfg.bin_size_ms,
        )
    elif cfg.model_type == "pca_latent_regression":
        preds = predict_pca_latent_regression(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            n_components=params["n_components"],
            ridge_alpha=params["ridge_alpha"],
        )
    elif cfg.model_type == "ridge_direct":
        preds = predict_ridge_direct(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            ridge_alpha=params["ridge_alpha"],
        )
    elif cfg.model_type == "lagged_ridge_direct":
        preds = predict_lagged_ridge_direct(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            ridge_alpha=params["ridge_alpha"],
            history_bins=params["history_bins"],
            input_transform=params["input_transform"],
        )
    elif cfg.model_type == "lagged_pca_latent_regression":
        preds = predict_lagged_pca_latent_regression(
            train_dict["train_spikes_heldin"],
            train_dict["train_spikes_heldout"],
            eval_dict["eval_spikes_heldin"],
            n_components=params["n_components"],
            ridge_alpha=params["ridge_alpha"],
            history_bins=params["history_bins"],
            input_transform=params["input_transform"],
        )
    else:
        raise ValueError(
            f"Unsupported model_type `{cfg.model_type}`. Expected one of "
            f"['smoothing', 'pca_latent_regression', 'ridge_direct', "
            f"'lagged_ridge_direct', 'lagged_pca_latent_regression']."
        )

    output_dict = {_dataset_key(cfg.dataset_name, cfg.bin_size_ms): preds}
    metrics = evaluate(target_dict, output_dict)[0][_split_key(cfg.dataset_name, cfg.bin_size_ms)]
    if not np.isfinite(metrics["co-bps"]):
        raise ValueError("co-bps is not finite; check preprocessing/model outputs")
    return output_dict, metrics


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


def _select_best_smoothing_params(dataset: NWBDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    imp = cfg.improvement
    folds = _build_cv_masks(dataset, cfg.train_split, imp["cv_folds"], cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for kern_sd, alpha in itertools.product(imp["kern_sd_grid"], imp["alpha_grid"]):
        fold_scores = []
        params = {"kern_sd_ms": kern_sd, "alpha": alpha, "log_offset": cfg.log_offset}
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=f"cv(kern_sd={kern_sd},alpha={alpha})",
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info(
            "CV candidate kern_sd=%s alpha=%s -> mean co-bps %.4f",
            kern_sd,
            alpha,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    logger.info(
        "Selected params for smoothing: kern_sd=%s alpha=%s (cv mean co-bps %.4f)",
        best_params["kern_sd_ms"],
        best_params["alpha"],
        best_score,
    )
    return best_params


def _select_best_pca_params(dataset: NWBDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    imp = cfg.improvement
    cv_folds = imp.get("cv_folds", 3)
    n_components_grid = imp.get("n_components_grid", [5, 10, 20, 30])
    ridge_alpha_grid = imp.get("ridge_alpha_grid", [1e-3, 1e-2, 1e-1, 1.0])
    folds = _build_cv_masks(dataset, cfg.train_split, cv_folds, cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for n_components, ridge_alpha in itertools.product(n_components_grid, ridge_alpha_grid):
        fold_scores = []
        params = {"n_components": int(n_components), "ridge_alpha": float(ridge_alpha)}
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=f"cv(n_components={n_components},ridge_alpha={ridge_alpha})",
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info(
            "CV candidate n_components=%s ridge_alpha=%s -> mean co-bps %.4f",
            n_components,
            ridge_alpha,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    logger.info(
        "Selected params for PCA latent regression: n_components=%s ridge_alpha=%s (cv mean co-bps %.4f)",
        best_params["n_components"],
        best_params["ridge_alpha"],
        best_score,
    )
    return best_params


def _select_best_ridge_direct_params(dataset: NWBDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    imp = cfg.improvement
    cv_folds = imp.get("cv_folds", 3)
    ridge_alpha_grid = imp.get("ridge_alpha_grid", [1e-3, 1e-2, 1e-1])
    folds = _build_cv_masks(dataset, cfg.train_split, cv_folds, cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for ridge_alpha in ridge_alpha_grid:
        fold_scores = []
        params = {"ridge_alpha": float(ridge_alpha)}
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=f"cv(ridge_alpha={ridge_alpha})",
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info(
            "CV candidate ridge_alpha=%s -> mean co-bps %.4f",
            ridge_alpha,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    logger.info(
        "Selected params for ridge direct: ridge_alpha=%s (cv mean co-bps %.4f)",
        best_params["ridge_alpha"],
        best_score,
    )
    return best_params


def _select_best_lagged_ridge_params(dataset: NWBDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    imp = cfg.improvement
    cv_folds = imp.get("cv_folds", 3)
    ridge_alpha_grid = imp.get("ridge_alpha_grid", [1e-3, 1e-2, 1e-1])
    history_bins_grid = imp.get("history_bins_grid", [3, 5, 9])
    input_transform = imp.get("input_transform", cfg.baseline.get("input_transform", "sqrt"))
    folds = _build_cv_masks(dataset, cfg.train_split, cv_folds, cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for history_bins, ridge_alpha in itertools.product(history_bins_grid, ridge_alpha_grid):
        fold_scores = []
        params = {
            "history_bins": int(history_bins),
            "ridge_alpha": float(ridge_alpha),
            "input_transform": input_transform,
        }
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=f"cv(history_bins={history_bins},ridge_alpha={ridge_alpha})",
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info(
            "CV candidate history_bins=%s ridge_alpha=%s input_transform=%s -> mean co-bps %.4f",
            history_bins,
            ridge_alpha,
            input_transform,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    logger.info(
        "Selected params for lagged ridge direct: history_bins=%s ridge_alpha=%s input_transform=%s (cv mean co-bps %.4f)",
        best_params["history_bins"],
        best_params["ridge_alpha"],
        best_params["input_transform"],
        best_score,
    )
    return best_params


def _select_best_lagged_pca_params(dataset: NWBDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    imp = cfg.improvement
    cv_folds = imp.get("cv_folds", 3)
    n_components_grid = imp.get("n_components_grid", [10, 20, 40, 80])
    ridge_alpha_grid = imp.get("ridge_alpha_grid", [1e-3, 1e-2, 1e-1, 1.0])
    history_bins_grid = imp.get("history_bins_grid", [3, 5, 9])
    input_transform = imp.get("input_transform", cfg.baseline.get("input_transform", "sqrt_zscore"))
    folds = _build_cv_masks(dataset, cfg.train_split, cv_folds, cfg.seed)

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for history_bins, n_components, ridge_alpha in itertools.product(
        history_bins_grid, n_components_grid, ridge_alpha_grid
    ):
        fold_scores = []
        params = {
            "history_bins": int(history_bins),
            "n_components": int(n_components),
            "ridge_alpha": float(ridge_alpha),
            "input_transform": input_transform,
        }
        for train_mask, eval_mask in folds:
            _, metrics = _run_single_eval(
                dataset,
                cfg,
                train_mask,
                eval_mask,
                params,
                include_psth=False,
                run_name=(
                    f"cv(history_bins={history_bins},n_components={n_components},"
                    f"ridge_alpha={ridge_alpha})"
                ),
            )
            fold_scores.append(metrics["co-bps"])
        mean_score = float(np.mean(fold_scores))
        logger.info(
            "CV candidate history_bins=%s n_components=%s ridge_alpha=%s input_transform=%s -> mean co-bps %.4f",
            history_bins,
            n_components,
            ridge_alpha,
            input_transform,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None
    logger.info(
        "Selected params for lagged PCA latent regression: history_bins=%s n_components=%s ridge_alpha=%s input_transform=%s (cv mean co-bps %.4f)",
        best_params["history_bins"],
        best_params["n_components"],
        best_params["ridge_alpha"],
        best_params["input_transform"],
        best_score,
    )
    return best_params


def run_full_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    set_seeds(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    pred_dir = ensure_dir(out_dir / "predictions")

    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(
        dataset_path,
        cfg.data_prefix,
        skip_fields=cfg.skip_fields,
    )
    dataset.resample(cfg.bin_size_ms)

    if cfg.model_type == "smoothing":
        reference_params: dict[str, Any] = {
            "kern_sd_ms": cfg.baseline["kern_sd_ms"],
            "alpha": cfg.baseline["alpha"],
            "log_offset": cfg.log_offset,
        }
        selected_params = _select_best_smoothing_params(dataset, cfg)
    elif cfg.model_type == "pca_latent_regression":
        reference_params = {
            "n_components": int(cfg.baseline.get("n_components", 10)),
            "ridge_alpha": float(cfg.baseline.get("ridge_alpha", 0.1)),
        }
        selected_params = _select_best_pca_params(dataset, cfg)
    elif cfg.model_type == "ridge_direct":
        reference_params = {"ridge_alpha": float(cfg.baseline.get("ridge_alpha", 0.1))}
        selected_params = _select_best_ridge_direct_params(dataset, cfg)
    elif cfg.model_type == "lagged_ridge_direct":
        reference_params = {
            "history_bins": int(cfg.baseline.get("history_bins", 5)),
            "ridge_alpha": float(cfg.baseline.get("ridge_alpha", 0.1)),
            "input_transform": cfg.baseline.get("input_transform", "sqrt"),
        }
        selected_params = _select_best_lagged_ridge_params(dataset, cfg)
    elif cfg.model_type == "lagged_pca_latent_regression":
        reference_params = {
            "history_bins": int(cfg.baseline.get("history_bins", 5)),
            "n_components": int(cfg.baseline.get("n_components", 20)),
            "ridge_alpha": float(cfg.baseline.get("ridge_alpha", 0.1)),
            "input_transform": cfg.baseline.get("input_transform", "sqrt_zscore"),
        }
        selected_params = _select_best_lagged_pca_params(dataset, cfg)
    else:
        raise ValueError(
            f"Unsupported model_type `{cfg.model_type}`. Expected one of "
            f"['smoothing', 'pca_latent_regression', 'ridge_direct', "
            f"'lagged_ridge_direct', 'lagged_pca_latent_regression']."
        )

    reference_output, reference_metrics = _run_single_eval(
        dataset,
        cfg,
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
    logger.info("Prediction artifact sha256 reference=%s selected=%s", reference_hash, selected_hash)
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

    repro = {
        "config": cfg.__dict__,
        "baseline_metrics": reference_metrics,
        "improved_metrics": selected_metrics,
        "baseline_params": reference_params,
        "improved_params": selected_params,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    return repro
