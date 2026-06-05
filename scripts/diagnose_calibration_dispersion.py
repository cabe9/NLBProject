"""Step 1.5: calibration and dispersion diagnostics on validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.nwb_interface import NWBDataset

from nlb_project.data_contract import resolve_data_path
from nlb_project.io_utils import ensure_dir

DEFAULT_SKIP_FIELDS = [
    "hand_pos",
    "cursor_pos",
    "eye_pos",
    "muscle_vel",
    "muscle_len",
    "joint_vel",
    "joint_ang",
    "force",
]

FOCUS_UNITS = (29, 43, 17)
N_CALIB_BINS = 12
MIN_BIN_SAMPLES = 500


def _dataset_key(dataset_name: str, bin_size_ms: int) -> str:
    return dataset_name if bin_size_ms == 5 else f"{dataset_name}_{bin_size_ms}"


def _load_target(
    dataset_name: str,
    data_path: str | None,
    data_prefix: str,
    bin_size_ms: int,
) -> np.ndarray:
    dataset_path = resolve_data_path(dataset_name, data_path, data_prefix)
    dataset = NWBDataset(dataset_path, data_prefix, skip_fields=DEFAULT_SKIP_FIELDS)
    dataset.resample(bin_size_ms)
    target = make_eval_target_tensors(
        dataset,
        dataset_name,
        train_trial_split="train",
        eval_trial_split="val",
        save_file=False,
        include_psth=False,
    )[dataset_name]
    return np.asarray(target["eval_spikes_heldout"], dtype=np.float64)


def _read_heldout_rates(path: Path, dataset_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[f"{dataset_key}/eval_rates_heldout"], dtype=np.float64)


def _unit_rate_groups(target: np.ndarray) -> dict[str, np.ndarray]:
    unit_spikes = np.sum(target, axis=(0, 1))
    low_cut = np.quantile(unit_spikes, 1 / 3)
    high_cut = np.quantile(unit_spikes, 2 / 3)
    return {
        "low_rate_units": unit_spikes <= low_cut,
        "mid_rate_units": (unit_spikes > low_cut) & (unit_spikes <= high_cut),
        "high_rate_units": unit_spikes > high_cut,
    }


def _subset(
    pred: np.ndarray, target: np.ndarray, unit_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if unit_mask is None:
        return pred.ravel(), target.ravel()
    return pred[:, :, unit_mask].ravel(), target[:, :, unit_mask].ravel()


def _calibration_rows(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    scope: str,
    n_bins: int = N_CALIB_BINS,
) -> list[dict[str, Any]]:
    pred_flat = np.clip(pred.ravel(), 0.0, None)
    target_flat = target.ravel()
    positive = pred_flat > 0
    if int(positive.sum()) < MIN_BIN_SAMPLES:
        pred_flat = pred_flat
    else:
        pred_flat = pred_flat[positive]
        target_flat = target_flat[positive]

    if pred_flat.size == 0:
        return []

    bin_edges = np.quantile(pred_flat, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 3:
        return []

    rows: list[dict[str, Any]] = []
    for idx in range(len(bin_edges) - 1):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (pred_flat >= lo) & (pred_flat <= hi)
        else:
            mask = (pred_flat >= lo) & (pred_flat < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(np.mean(pred_flat[mask]))
        mean_target = float(np.mean(target_flat[mask]))
        rel_error = (mean_target - mean_pred) / max(mean_target, 1e-8)
        rows.append(
            {
                "scope": scope,
                "bin": idx,
                "pred_lo": lo,
                "pred_hi": hi,
                "n_samples": count,
                "mean_pred": mean_pred,
                "mean_target": mean_target,
                "calibration_error": mean_target - mean_pred,
                "relative_error": rel_error,
            }
        )
    return rows


def _dispersion_rows(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    scope: str,
    n_bins: int = N_CALIB_BINS,
) -> list[dict[str, Any]]:
    pred_flat = np.clip(pred.ravel(), 0.0, None)
    target_flat = target.ravel()
    if pred_flat.size == 0:
        return []

    bin_edges = np.quantile(pred_flat, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 3:
        return []

    rows: list[dict[str, Any]] = []
    for idx in range(len(bin_edges) - 1):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (pred_flat >= lo) & (pred_flat <= hi)
        else:
            mask = (pred_flat >= lo) & (pred_flat < hi)
        count = int(mask.sum())
        if count < MIN_BIN_SAMPLES:
            continue
        bin_pred = pred_flat[mask]
        bin_target = target_flat[mask]
        mean_pred = float(np.mean(bin_pred))
        var_target = float(np.var(bin_target))
        poisson_var = mean_pred
        dispersion_ratio = var_target / max(poisson_var, 1e-8)
        rows.append(
            {
                "scope": scope,
                "bin": idx,
                "pred_lo": lo,
                "pred_hi": hi,
                "n_samples": count,
                "mean_pred": mean_pred,
                "var_target": var_target,
                "poisson_var": poisson_var,
                "dispersion_ratio": dispersion_ratio,
                "excess_zeros_frac": float(np.mean(bin_target == 0)),
            }
        )
    return rows


def _evaluate_gates(
    calibration: pd.DataFrame,
    dispersion: pd.DataFrame,
) -> dict[str, Any]:
    overall_cal = calibration[calibration["scope"] == "overall"]
    overall_disp = dispersion[dispersion["scope"] == "overall"]

    cal_max_abs = (
        float(overall_cal["calibration_error"].abs().max()) if not overall_cal.empty else 0.0
    )
    cal_mean_abs = (
        float(overall_cal["calibration_error"].abs().mean()) if not overall_cal.empty else 0.0
    )
    cal_bins_over_5pct = (
        int((overall_cal["relative_error"].abs() >= 0.05).sum()) if not overall_cal.empty else 0
    )

    focus_cal = calibration[calibration["scope"].str.startswith("unit_")]
    focus_max_abs = (
        float(focus_cal["calibration_error"].abs().max()) if not focus_cal.empty else 0.0
    )

    disp_ratios = (
        overall_disp["dispersion_ratio"].to_numpy() if not overall_disp.empty else np.array([])
    )
    disp_median = float(np.median(disp_ratios)) if disp_ratios.size else 1.0
    disp_bins_over_125 = int(np.sum(disp_ratios >= 1.25)) if disp_ratios.size else 0
    disp_bins_over_150 = int(np.sum(disp_ratios >= 1.50)) if disp_ratios.size else 0

    calibration_signal = bool(
        cal_bins_over_5pct >= 3 or cal_max_abs >= 0.002 or focus_max_abs >= 0.003
    )
    dispersion_signal = bool(
        disp_ratios.size >= 3 and disp_bins_over_125 >= max(2, int(np.ceil(0.5 * disp_ratios.size)))
    ) or bool(disp_median >= 1.35)

    if calibration_signal and not dispersion_signal:
        recommendation = "calibration_adapter_or_unit_scale_bias"
    elif dispersion_signal and not calibration_signal:
        recommendation = "negative_binomial_readout"
    elif calibration_signal and dispersion_signal:
        recommendation = "calibration_first_then_nb_if_still_short"
    else:
        recommendation = "stop_no_structural_lever"

    return {
        "calibration_signal": calibration_signal,
        "dispersion_signal": dispersion_signal,
        "cal_max_abs_error": cal_max_abs,
        "cal_mean_abs_error": cal_mean_abs,
        "cal_bins_over_5pct_rel": cal_bins_over_5pct,
        "focus_units_max_abs_error": focus_max_abs,
        "dispersion_median_ratio": disp_median,
        "dispersion_bins_over_1_25": disp_bins_over_125,
        "dispersion_bins_over_1_50": disp_bins_over_150,
        "recommendation": recommendation,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_summary(
    out_dir: Path,
    calibration: pd.DataFrame,
    dispersion: pd.DataFrame,
    gate: dict[str, Any],
    model_name: str,
) -> None:
    rec = gate["recommendation"]
    lines = [
        "# Step 1.5: Calibration and Dispersion Diagnostic",
        "",
        f"Model: `{model_name}`.",
        "",
        "## Recommendation",
        "",
        f"**{rec}**",
        "",
        f"- Calibration signal: {'yes' if gate['calibration_signal'] else 'no'}",
        f"- Dispersion signal: {'yes' if gate['dispersion_signal'] else 'no'}",
        f"- Max |calibration error| (overall bins): {gate['cal_max_abs_error']:.6f}",
        f"- Max |calibration error| (units 29/43/17): {gate['focus_units_max_abs_error']:.6f}",
        f"- Median dispersion ratio (var/mean): {gate['dispersion_median_ratio']:.3f}",
        f"- Bins with ratio >= 1.25: {gate['dispersion_bins_over_1_25']}",
        "",
        "Interpretation:",
        "- `calibration_error` = mean_target - mean_pred (positive = under-prediction).",
        "- `dispersion_ratio` = Var(target) / mean(pred); Poisson expects ~1.0.",
        "- If calibration only → tiny adapter or unit scale/bias.",
        "- If dispersion only → negative-binomial readout (not ZINB as first test).",
        "- If neither → stop cleanly.",
        "",
        "## Calibration Curves (overall)",
        "",
        _markdown_table(calibration[calibration["scope"] == "overall"]),
        "",
        "## Calibration Curves (unit rate groups)",
        "",
        _markdown_table(
            calibration[calibration["scope"].str.endswith("_rate_units")].sort_values(
                ["scope", "bin"]
            )
        ),
        "",
        "## Calibration Curves (units 29, 43, 17)",
        "",
        _markdown_table(
            calibration[calibration["scope"].str.startswith("unit_")].sort_values(["scope", "bin"])
        ),
        "",
        "## Dispersion Check (overall)",
        "",
        _markdown_table(dispersion[dispersion["scope"] == "overall"]),
        "",
        "## Dispersion Check (unit rate groups)",
        "",
        _markdown_table(
            dispersion[dispersion["scope"].str.endswith("_rate_units")].sort_values(
                ["scope", "bin"]
            )
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "go_no_go.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration and dispersion diagnostics.")
    parser.add_argument(
        "--output-dir",
        default="results/diagnostics/headline_calibration_dispersion",
    )
    parser.add_argument("--dataset-name", default="mc_maze")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-prefix", default="*full")
    parser.add_argument("--bin-size-ms", type=int, default=5)
    parser.add_argument("--model-name", default="headline_lr0013_depth5")
    parser.add_argument(
        "--prediction",
        default=(
            "results/benchmark_runs/stndt_lite_diverse_ensemble_screen/"
            "predictions/improved_predictions.h5"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    dataset_key = _dataset_key(args.dataset_name, args.bin_size_ms)
    target = _load_target(args.dataset_name, args.data_path, args.data_prefix, args.bin_size_ms)
    pred = _read_heldout_rates(Path(args.prediction), dataset_key)
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} != target {target.shape}")

    cal_rows: list[dict[str, Any]] = []
    disp_rows: list[dict[str, Any]] = []

    cal_rows.extend(
        _calibration_rows(pred, target, scope="overall"),
    )
    disp_rows.extend(
        _dispersion_rows(pred, target, scope="overall"),
    )

    for group_name, unit_mask in _unit_rate_groups(target).items():
        p_sub, t_sub = _subset(pred, target, unit_mask)
        cal_rows.extend(_calibration_rows(p_sub, t_sub, scope=group_name))
        disp_rows.extend(_dispersion_rows(p_sub, t_sub, scope=group_name))

    for unit_idx in FOCUS_UNITS:
        cal_rows.extend(
            _calibration_rows(
                pred[:, :, unit_idx], target[:, :, unit_idx], scope=f"unit_{unit_idx}"
            )
        )
        disp_rows.extend(
            _dispersion_rows(pred[:, :, unit_idx], target[:, :, unit_idx], scope=f"unit_{unit_idx}")
        )

    calibration = pd.DataFrame(cal_rows)
    dispersion = pd.DataFrame(disp_rows)
    calibration.to_csv(out_dir / "calibration_curves.csv", index=False)
    dispersion.to_csv(out_dir / "dispersion_check.csv", index=False)

    gate = _evaluate_gates(calibration, dispersion)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model_name,
                "prediction": args.prediction,
                "dataset_name": args.dataset_name,
                "focus_units": list(FOCUS_UNITS),
                "gate": gate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary(out_dir, calibration, dispersion, gate, args.model_name)
    print(f"Wrote calibration/dispersion diagnostic -> {out_dir}")
    print(f"Recommendation: {gate['recommendation']}")


if __name__ == "__main__":
    main()
