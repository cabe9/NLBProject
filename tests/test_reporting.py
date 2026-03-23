from __future__ import annotations

import csv
from pathlib import Path

from nlb_project.reporting import (
    ComparisonSpec,
    build_comparison_rows,
    write_comparison_md,
    write_comparison_svg,
    write_metric_panel_svg,
)


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "model_type", "co-bps", "vel R2", "psth R2", "params"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_comparison_rows_and_outputs(tmp_path: Path) -> None:
    _write_metrics(
        tmp_path / "results/benchmark_runs/static_pca/metrics.csv",
        [
            {
                "model": "baseline",
                "model_type": "pca_latent_regression",
                "co-bps": "0.01",
                "vel R2": "0.2",
                "psth R2": "",
                "params": '{"n_components": 10, "ridge_alpha": 0.1}',
            }
        ],
    )
    _write_metrics(
        tmp_path / "results/benchmark_runs/lagged_pca/metrics.csv",
        [
            {
                "model": "improved",
                "model_type": "lagged_pca_latent_regression",
                "co-bps": "0.05",
                "vel R2": "0.4",
                "psth R2": "",
                "params": '{"history_bins": 9, "n_components": 20, "ridge_alpha": 0.1, "input_transform": "sqrt_zscore"}',
            }
        ],
    )
    specs = [
        ComparisonSpec(
            label="static PCA latent regression",
            metrics_path="results/benchmark_runs/static_pca/metrics.csv",
            model_row="baseline",
            note="static",
        ),
        ComparisonSpec(
            label="lagged PCA latent regression (selected history)",
            metrics_path="results/benchmark_runs/lagged_pca/metrics.csv",
            model_row="improved",
            note="lagged",
        ),
    ]

    rows = build_comparison_rows(tmp_path, specs)

    assert rows[0]["model_type"] == "pca_latent_regression"
    assert rows[0]["role"] == "reference"
    assert rows[1]["history_bins"] == 9
    assert rows[1]["role"] == "selected"

    md_path = tmp_path / "comparison.md"
    svg_path = tmp_path / "comparison.svg"
    panel_path = tmp_path / "comparison_panel.svg"
    write_comparison_md(rows, md_path)
    write_comparison_svg(rows, svg_path)
    write_metric_panel_svg(rows, panel_path)

    assert "static PCA latent regression" in md_path.read_text(encoding="utf-8")
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    assert "vel R2" in panel_path.read_text(encoding="utf-8")
