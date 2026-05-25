from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PredictionSet:
    name: str
    path: Path


def _read_heldout_rates(path: Path, dataset_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[f"{dataset_key}/eval_rates_heldout"], dtype=np.float64)


def _poisson_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 1e-8, None)
    return pred - target * np.log(pred)


def _phase_masks(n_bins: int) -> dict[str, np.ndarray]:
    bins = np.arange(n_bins)
    return {
        "pre_move_-250_0ms": bins < 50,
        "early_move_0_200ms": (bins >= 50) & (bins < 90),
        "late_move_200_450ms": bins >= 90,
    }


def _load_target(
    dataset_name: str,
    data_path: str | None,
    data_prefix: str,
    bin_size_ms: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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
    val_info = dataset.trial_info[dataset.trial_info.split == "val"].reset_index(drop=True)
    return (
        np.asarray(target["eval_spikes_heldout"], dtype=np.float64),
        np.asarray(target["eval_behavior"], dtype=np.float64),
        val_info,
    )


def _model_summary_rows(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    behavior: np.ndarray,
) -> list[dict[str, Any]]:
    phase_masks = _phase_masks(target.shape[1])
    rows: list[dict[str, Any]] = []
    losses = {name: _poisson_loss(pred, target) for name, pred in predictions.items()}
    reference_name = next(iter(predictions))
    reference_loss = losses[reference_name]
    speed = np.linalg.norm(behavior, axis=-1)
    speed_thresholds = {
        "low_speed": speed <= np.nanquantile(speed, 1 / 3),
        "mid_speed": (speed > np.nanquantile(speed, 1 / 3))
        & (speed <= np.nanquantile(speed, 2 / 3)),
        "high_speed": speed > np.nanquantile(speed, 2 / 3),
    }

    for name, loss in losses.items():
        delta = reference_loss - loss
        rows.append(
            {
                "model": name,
                "slice": "overall",
                "mean_loss": float(np.mean(loss)),
                "delta_vs_reference": float(np.mean(delta)),
                "target_spikes": float(np.sum(target)),
            }
        )
        for phase, mask in phase_masks.items():
            rows.append(
                {
                    "model": name,
                    "slice": phase,
                    "mean_loss": float(np.mean(loss[:, mask, :])),
                    "delta_vs_reference": float(np.mean(delta[:, mask, :])),
                    "target_spikes": float(np.sum(target[:, mask, :])),
                }
            )
        for bucket, mask in speed_thresholds.items():
            expanded = mask[:, :, None]
            rows.append(
                {
                    "model": name,
                    "slice": bucket,
                    "mean_loss": float(np.mean(loss[expanded.repeat(target.shape[2], axis=2)])),
                    "delta_vs_reference": float(
                        np.mean(delta[expanded.repeat(target.shape[2], axis=2)])
                    ),
                    "target_spikes": float(
                        np.sum(target[expanded.repeat(target.shape[2], axis=2)])
                    ),
                }
            )
    return rows


def _unit_rows(predictions: dict[str, np.ndarray], target: np.ndarray) -> list[dict[str, Any]]:
    losses = {name: _poisson_loss(pred, target) for name, pred in predictions.items()}
    reference_name = next(iter(predictions))
    reference_loss = losses[reference_name]
    rows: list[dict[str, Any]] = []
    for name, loss in losses.items():
        delta = np.mean(reference_loss - loss, axis=(0, 1))
        mean_loss = np.mean(loss, axis=(0, 1))
        spikes = np.sum(target, axis=(0, 1))
        for unit_idx in range(target.shape[2]):
            rows.append(
                {
                    "model": name,
                    "unit": unit_idx,
                    "mean_loss": float(mean_loss[unit_idx]),
                    "delta_vs_reference": float(delta[unit_idx]),
                    "target_spikes": float(spikes[unit_idx]),
                }
            )
    return rows


def _condition_rows(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    val_info: pd.DataFrame,
    field: str,
) -> list[dict[str, Any]]:
    losses = {name: _poisson_loss(pred, target) for name, pred in predictions.items()}
    reference_name = next(iter(predictions))
    reference_loss = losses[reference_name]
    rows: list[dict[str, Any]] = []
    for value in sorted(val_info[field].dropna().unique()):
        trial_mask = val_info[field].to_numpy() == value
        if int(np.sum(trial_mask)) < 3:
            continue
        for name, loss in losses.items():
            delta = reference_loss - loss
            rows.append(
                {
                    "field": field,
                    "value": value,
                    "model": name,
                    "n_trials": int(np.sum(trial_mask)),
                    "mean_loss": float(np.mean(loss[trial_mask])),
                    "delta_vs_reference": float(np.mean(delta[trial_mask])),
                    "target_spikes": float(np.sum(target[trial_mask])),
                }
            )
    return rows


def _write_summary(
    out_dir: Path,
    model_summary: pd.DataFrame,
    unit_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    reference_name: str,
) -> None:
    def markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
        shown = frame if max_rows is None else frame.head(max_rows)
        columns = list(shown.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in shown.itertuples(index=False, name=None):
            cells = []
            for value in row:
                if isinstance(value, float):
                    cells.append(f"{value:.6f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    overall = model_summary[model_summary["slice"] == "overall"].copy()
    overall = overall.sort_values("delta_vs_reference", ascending=False)
    top_units = (
        unit_summary[unit_summary["model"] != reference_name]
        .sort_values("delta_vs_reference", ascending=False)
        .head(12)
    )
    weak_units = (
        unit_summary[unit_summary["model"] != reference_name]
        .sort_values("delta_vs_reference", ascending=True)
        .head(12)
    )
    top_conditions = (
        condition_summary[condition_summary["model"] != reference_name]
        .sort_values("delta_vs_reference", ascending=False)
        .head(12)
    )

    lines = [
        "# Validation Residual Diagnostic",
        "",
        f"Reference prediction set: `{reference_name}`.",
        "Positive `delta_vs_reference` means lower Poisson residual loss than the reference.",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## Best Unit-Level Improvements",
        "",
        markdown_table(top_units),
        "",
        "## Largest Unit-Level Regressions",
        "",
        markdown_table(weak_units),
        "",
        "## Best Condition-Level Improvements",
        "",
        markdown_table(top_conditions),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose validation residual patterns.")
    parser.add_argument("--output-dir", default="results/diagnostics/validation_residuals")
    parser.add_argument("--dataset-name", default="mc_maze")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-prefix", default="*full")
    parser.add_argument("--bin-size-ms", type=int, default=5)
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Prediction label and HDF5 path. First prediction is the reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    target, behavior, val_info = _load_target(
        args.dataset_name,
        args.data_path,
        args.data_prefix,
        args.bin_size_ms,
    )
    prediction_sets = [
        PredictionSet(name=name, path=Path(path)) for name, path in args.prediction
    ]
    predictions = {
        item.name: _read_heldout_rates(item.path, args.dataset_name) for item in prediction_sets
    }
    for name, pred in predictions.items():
        if pred.shape != target.shape:
            raise ValueError(f"{name}: prediction shape {pred.shape} != target {target.shape}")

    model_summary = pd.DataFrame(_model_summary_rows(predictions, target, behavior))
    unit_summary = pd.DataFrame(_unit_rows(predictions, target))
    condition_summary = pd.concat(
        [
            pd.DataFrame(_condition_rows(predictions, target, val_info, "trial_type")),
            pd.DataFrame(_condition_rows(predictions, target, val_info, "maze_id")),
        ],
        ignore_index=True,
    )
    model_summary.to_csv(out_dir / "model_slices.csv", index=False)
    unit_summary.to_csv(out_dir / "unit_slices.csv", index=False)
    condition_summary.to_csv(out_dir / "condition_slices.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "bin_size_ms": args.bin_size_ms,
                "reference": prediction_sets[0].name,
                "predictions": [
                    {"name": item.name, "path": str(item.path)} for item in prediction_sets
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary(out_dir, model_summary, unit_summary, condition_summary, prediction_sets[0].name)
    print(f"Wrote residual diagnostic -> {out_dir}")


if __name__ == "__main__":
    main()
