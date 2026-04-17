"""Compare co-bps across three output heads on the ``mc_maze`` train/val split.

Runs every model in the tracked comparison set with the hyperparameters that
produced the checked-in ``results/benchmark_runs/*/metrics.csv`` numbers under:

* ``linear``     - legacy Gaussian-ridge-on-counts readout (baseline).
* ``log_link``   - Gaussian ridge on ``log(count + offset)`` with Duan's
  smearing correction (fast, strictly positive).
* ``poisson_glm`` - per-neuron ``sklearn.linear_model.PoissonRegressor``;
  the co-bps-correct readout. Slower but fit under the same likelihood the
  metric scores, so it removes the Jensen / smearing approximation of
  ``log_link``.

Prints a 3-head side-by-side table and optionally dumps JSON. Does not write
to ``results/benchmark_runs/`` so tracked artifacts stay intact.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep imports after argparse so --help works without loading heavy deps.
from nlb_project.config import ExperimentConfig
from nlb_project.data_contract import resolve_data_path
from nlb_project.pipeline import _run_single_eval  # type: ignore[import]

logger = logging.getLogger("compare_output_heads")


@dataclass(frozen=True)
class Row:
    """One entry in the tracked-comparison manifest."""

    label: str
    model_type: str
    params: dict[str, Any]
    old_cobps: float
    old_vel_r2: float
    old_source: str


# Pulled straight from the checked-in metrics.csv files so the old column is
# the published number, not something regenerated here.
ROWS: list[Row] = [
    Row(
        label="static PCA latent regression",
        model_type="pca_latent_regression",
        params={"n_components": 10, "ridge_alpha": 0.1},
        old_cobps=0.0038680972481445605,
        old_vel_r2=0.07552087936814766,
        old_source="results/benchmark_runs/static_pca/metrics.csv",
    ),
    Row(
        label="static direct ridge",
        model_type="ridge_direct",
        params={"ridge_alpha": 0.1},
        old_cobps=-0.0334800021487025,
        old_vel_r2=0.07614880872615087,
        old_source="results/benchmark_runs/static_ridge/metrics.csv",
    ),
    Row(
        label="lagged direct ridge (5 bins)",
        model_type="lagged_ridge_direct",
        params={"history_bins": 5, "ridge_alpha": 0.1, "input_transform": "sqrt"},
        old_cobps=-0.430059038486296,
        old_vel_r2=0.20073108645832333,
        old_source="results/benchmark_runs/lagged_ridge_single/metrics.csv",
    ),
    Row(
        label="lagged reduced-rank regression (selected)",
        model_type="lagged_reduced_rank_regression",
        params={
            "history_bins": 5,
            "rank": 5,
            "ridge_alpha": 0.1,
            "input_transform": "sqrt_zscore",
        },
        old_cobps=-0.009057627433083389,
        old_vel_r2=0.15944457408890028,
        old_source="results/benchmark_runs/lagged_rrr_sweep/metrics.csv",
    ),
    Row(
        label="lagged PCA latent regression (5 bins)",
        model_type="lagged_pca_latent_regression",
        params={
            "history_bins": 5,
            "n_components": 20,
            "ridge_alpha": 0.1,
            "input_transform": "sqrt_zscore",
        },
        old_cobps=0.04176870805106905,
        old_vel_r2=0.2441094526839589,
        old_source="results/benchmark_runs/lagged_pca_single/metrics.csv",
    ),
    Row(
        label="lagged PCA latent regression (selected history)",
        model_type="lagged_pca_latent_regression",
        params={
            "history_bins": 9,
            "n_components": 20,
            "ridge_alpha": 0.1,
            "input_transform": "sqrt_zscore",
        },
        old_cobps=0.04859667651088111,
        old_vel_r2=0.3730257369981418,
        old_source="results/benchmark_runs/lagged_pca_history_sweep/metrics.csv",
    ),
]


def _build_cfg(row: Row, data_path: str) -> ExperimentConfig:
    """Build a minimal ExperimentConfig matching the original benchmark setup."""
    return ExperimentConfig(
        dataset_name="mc_maze",
        data_path=data_path,
        data_prefix="*full",
        bin_size_ms=5,
        train_split="train",
        eval_split="val",
        include_psth=False,
        log_offset=0.0001,
        seed=0,
        skip_fields=[
            "hand_pos",
            "cursor_pos",
            "eye_pos",
            "muscle_vel",
            "muscle_len",
            "joint_vel",
            "joint_ang",
            "force",
        ],
        baseline={},
        improvement={},
        output_dir="",
        model_type=row.model_type,
    )


def _run_with_head(
    dataset,
    cfg: ExperimentConfig,
    row: Row,
    head: str,
    log_offset: float,
) -> dict[str, float]:
    params = dict(row.params)
    params["output_head"] = head
    params["log_offset"] = log_offset
    t0 = time.perf_counter()
    _, metrics = _run_single_eval(
        dataset,
        cfg,
        cfg.train_split,
        cfg.eval_split,
        params,
        include_psth=False,
        run_name=f"{row.label}[{head}]",
    )
    return {
        "co-bps": float(metrics.get("co-bps", float("nan"))),
        "vel R2": float(metrics.get("vel R2", float("nan"))),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/raw/000128/sub-Jenkins",
        help="Path containing the mc_maze NWB files.",
    )
    parser.add_argument(
        "--log-offset",
        type=float,
        default=1e-3,
        help="log_offset for the log_link head (default: 1e-3).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to dump the comparison as JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    # Import NWBDataset lazily so --help stays fast.
    from nlb_tools.nwb_interface import NWBDataset  # type: ignore[import]

    dataset_path = resolve_data_path("mc_maze", args.data_path, "*full")
    print(f"Loading NWB dataset from {dataset_path} ...")
    dataset = NWBDataset(
        dataset_path,
        "*full",
        skip_fields=[
            "hand_pos",
            "cursor_pos",
            "eye_pos",
            "muscle_vel",
            "muscle_len",
            "joint_vel",
            "joint_ang",
            "force",
        ],
    )
    dataset.resample(5)
    print("Dataset loaded. Running comparisons ...\n")

    results: list[dict[str, Any]] = []
    for row in ROWS:
        cfg = _build_cfg(row, data_path=args.data_path)
        linear = _run_with_head(dataset, cfg, row, "linear", args.log_offset)
        loglink = _run_with_head(dataset, cfg, row, "log_link", args.log_offset)
        poisson = _run_with_head(dataset, cfg, row, "poisson_glm", args.log_offset)
        results.append(
            {
                "label": row.label,
                "model_type": row.model_type,
                "params": row.params,
                "old_cobps": row.old_cobps,
                "old_vel_r2": row.old_vel_r2,
                "old_source": row.old_source,
                "linear_cobps": linear["co-bps"],
                "linear_vel_r2": linear["vel R2"],
                "linear_wall_s": linear["wall_s"],
                "log_link_cobps": loglink["co-bps"],
                "log_link_vel_r2": loglink["vel R2"],
                "log_link_wall_s": loglink["wall_s"],
                "poisson_glm_cobps": poisson["co-bps"],
                "poisson_glm_vel_r2": poisson["vel R2"],
                "poisson_glm_wall_s": poisson["wall_s"],
                "delta_cobps_log_link_vs_linear": loglink["co-bps"] - linear["co-bps"],
                "delta_cobps_poisson_vs_linear": poisson["co-bps"] - linear["co-bps"],
                "delta_cobps_poisson_vs_log_link": poisson["co-bps"] - loglink["co-bps"],
                "delta_cobps_poisson_vs_tracked": poisson["co-bps"] - row.old_cobps,
            }
        )

    _print_markdown(results)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        print(f"\nWrote JSON comparison to {args.json_out}")


def _fmt(x: float) -> str:
    return f"{x: .4f}" if np.isfinite(x) else "   nan"


def _print_markdown(results: list[dict[str, Any]]) -> None:
    print("# co-bps: `linear` vs `log_link` vs `poisson_glm` output heads")
    print()
    print(
        "| model | old (tracked) | linear (rerun) | log_link | poisson_glm "
        "| Δ log_link-linear | Δ poisson-linear | Δ poisson-log_link |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['label']} "
            f"| {_fmt(r['old_cobps'])} "
            f"| {_fmt(r['linear_cobps'])} "
            f"| {_fmt(r['log_link_cobps'])} "
            f"| {_fmt(r['poisson_glm_cobps'])} "
            f"| {_fmt(r['delta_cobps_log_link_vs_linear'])} "
            f"| {_fmt(r['delta_cobps_poisson_vs_linear'])} "
            f"| {_fmt(r['delta_cobps_poisson_vs_log_link'])} |"
        )
    print()
    print("Wall-clock (seconds per model, single fit on mc_maze train/val):")
    print("| model | linear | log_link | poisson_glm |")
    print("|---|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['label']} "
            f"| {r['linear_wall_s']:6.1f} "
            f"| {r['log_link_wall_s']:6.1f} "
            f"| {r['poisson_glm_wall_s']:6.1f} |"
        )
    print()
    print("Notes:")
    print("- `old (tracked)` = value in results/benchmark_runs/<run>/metrics.csv.")
    print(
        "- `linear` = legacy Gaussian-ridge-on-counts head, clipped to [1e-9, 1e20]."
    )
    print(
        "- `log_link` = Gaussian ridge on log(count + offset) + Duan smearing. Fast, "
        "strictly positive, but mis-specified for Poisson likelihood."
    )
    print(
        "- `poisson_glm` = per-neuron sklearn PoissonRegressor. Fit under the "
        "exact likelihood co-bps scores."
    )
    print("- Positive numbers are better; co-bps is the primary NLB metric.")


if __name__ == "__main__":
    main()
