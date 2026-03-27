"""Utilities for portfolio-facing result artifacts.

The metrics themselves are always loaded from saved ``metrics.csv`` files.
The only manual layer is the small manifest below, which defines:
- which saved run directories belong in the public comparison
- which row to read from each metrics file (for example ``baseline`` vs ``improved``)
- the short human-facing label and note for that row
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    metrics_path: str
    model_row: str
    note: str


DEFAULT_COMPARISON_SPECS = [
    ComparisonSpec(
        label="static PCA latent regression",
        metrics_path="results/benchmark_runs/static_pca/metrics.csv",
        model_row="baseline",
        note="Static latent baseline; no temporal context.",
    ),
    ComparisonSpec(
        label="static direct ridge",
        metrics_path="results/benchmark_runs/static_ridge/metrics.csv",
        model_row="baseline",
        note="Direct one-bin regression is not competitive.",
    ),
    ComparisonSpec(
        label="lagged direct ridge (5 bins)",
        metrics_path="results/benchmark_runs/lagged_ridge_single/metrics.csv",
        model_row="baseline",
        note="Temporal history alone overfit without a latent bottleneck.",
    ),
    ComparisonSpec(
        label="lagged reduced-rank regression (selected)",
        metrics_path="results/benchmark_runs/lagged_rrr_sweep/metrics.csv",
        model_row="improved",
        note="A supervised low-rank mapping stayed worse than lagged PCA on co-bps.",
    ),
    ComparisonSpec(
        label="lagged PCA latent regression (5 bins)",
        metrics_path="results/benchmark_runs/lagged_pca_single/metrics.csv",
        model_row="baseline",
        note="Temporal context plus train-only conditioning gave the first real gain.",
    ),
    ComparisonSpec(
        label="lagged PCA latent regression (selected history)",
        metrics_path="results/benchmark_runs/lagged_pca_history_sweep/metrics.csv",
        model_row="improved",
        note="Best validated model in the repo.",
    ),
]


def _load_metrics_row(metrics_path: Path, model_row: str) -> dict[str, str]:
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    for row in rows:
        if row["model"] == model_row:
            return row
    raise ValueError(f"Could not find row `{model_row}` in {metrics_path}")


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def build_comparison_rows(
    root: str | Path,
    specs: list[ComparisonSpec] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for spec in specs or DEFAULT_COMPARISON_SPECS:
        metrics_path = root / spec.metrics_path
        metric_row = _load_metrics_row(metrics_path, spec.model_row)
        params = json.loads(metric_row["params"])
        rows.append(
            {
                "model_label": spec.label,
                "model_type": metric_row["model_type"],
                "role": "reference" if spec.model_row == "baseline" else "selected",
                "history_bins": params.get("history_bins", ""),
                "rank": params.get("rank", ""),
                "n_components": params.get("n_components", ""),
                "ridge_alpha": params.get("ridge_alpha", ""),
                "input_transform": params.get("input_transform", "none"),
                "co_bps": float(metric_row["co-bps"]),
                "vel_r2": _parse_float(metric_row.get("vel R2")),
                "psth_r2": _parse_float(metric_row.get("psth R2")),
                "source_metrics_path": spec.metrics_path,
                "source_row": spec.model_row,
                "scientific_note": spec.note,
            }
        )
    return rows


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_comparison_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label",
        "model_type",
        "role",
        "history_bins",
        "rank",
        "n_components",
        "ridge_alpha",
        "input_transform",
        "co_bps",
        "vel_r2",
        "psth_r2",
        "source_metrics_path",
        "source_row",
        "scientific_note",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_md(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    lines = [
        "# Model Comparison",
        "",
        "Generated from saved `metrics.csv` artifacts in `results/benchmark_runs/*/`.",
        "The only manual part is the comparison manifest in `src/nlb_project/reporting.py`, which selects which saved run rows to display.",
        "",
        "| model | role | history bins | rank | n_components | ridge_alpha | transform | co-bps | vel R2 | source |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model_label} | {role} | {history_bins} | {rank} | {n_components} | {ridge_alpha} | "
            "{input_transform} | {co_bps} | {vel_r2} | `{source_metrics_path}` ({source_row}) |".format(
                model_label=row["model_label"],
                role=row["role"],
                history_bins=row["history_bins"] or "n/a",
                rank=row["rank"] or "n/a",
                n_components=row["n_components"] or "n/a",
                ridge_alpha=row["ridge_alpha"] or "n/a",
                input_transform=row["input_transform"] or "none",
                co_bps=_fmt_num(row["co_bps"]),
                vel_r2=_fmt_num(row["vel_r2"]),
                source_metrics_path=row["source_metrics_path"],
                source_row=row["source_row"],
            )
        )
    lines.extend(
        [
            "",
            "Takeaway:",
            "- The winning change was adding short neural history and compressing that history before regression.",
            "- Temporal context mattered more than static latent dimensionality alone.",
            "- A supervised reduced-rank mapping did not recover the same co-smoothing gain as lagged PCA.",
        ]
    )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_experiment_log_md(
    rows: list[dict[str, Any]],
    main_metrics_path: str | Path,
    out_path: str | Path,
) -> None:
    main_rows = list(csv.DictReader(Path(main_metrics_path).open(encoding="utf-8")))
    reference = next(row for row in main_rows if row["model"] == "baseline")
    selected = next(row for row in main_rows if row["model"] == "improved")
    delta = float(selected["co-bps"]) - float(reference["co-bps"])

    lines = [
        "# Experiment Log",
        "",
        "This summary is generated from committed `metrics.csv` artifacts.",
        "",
        "## Main comparison",
        "",
        "| model | co-bps | vel R2 | note |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {_fmt_num(row['co_bps'])} | {_fmt_num(row['vel_r2'])} | {row['scientific_note']} |"
        )
    lines.extend(
        [
            "",
            "## Validated lagged PCA result",
            "",
            f"- reference co-bps: `{float(reference['co-bps']):.4f}`",
            f"- selected co-bps: `{float(selected['co-bps']):.4f}`",
            f"- delta co-bps: `{delta:.4f}`",
            "",
            "## Interpretation",
            "",
            "- The original static PCA model was weak because it ignored short-timescale neural history.",
            "- Direct lagged ridge showed that temporal context alone is not enough; the lagged design needs compression.",
            "- Lagged PCA latent regression was the first model family that improved co-smoothing substantially while remaining simple and interpretable.",
            "- Lagged reduced-rank regression did not beat lagged PCA, which suggests the PCA bottleneck was already a better fit than this simple supervised low-rank mapping.",
        ]
    )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_comparison_svg(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    width = 980
    left = 320
    right = 70
    top = 40
    row_h = 56
    bar_h = 24
    bottom = 50
    height = top + bottom + row_h * len(rows)

    values = [float(row["co_bps"]) for row in rows]
    min_val = min(0.0, min(values))
    max_val = max(values)
    span = max_val - min_val
    pad = span * 0.1 if span > 0 else 0.1
    min_plot = min_val - pad
    max_plot = max_val + pad
    plot_w = width - left - right

    def x_pos(value: float) -> float:
        return left + ((value - min_plot) / (max_plot - min_plot)) * plot_w

    zero_x = x_pos(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, Helvetica, sans-serif; fill: #1b1f24; }',
        '.title { font-size: 24px; font-weight: 700; }',
        '.label { font-size: 14px; }',
        '.value { font-size: 13px; font-weight: 600; }',
        '.axis { stroke: #9aa4af; stroke-width: 1; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="30" y="30">mc_maze co-bps comparison</text>',
    ]

    for tick in [min_plot, 0.0, max_plot]:
        x = x_pos(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 10}"/>')
        parts.append(f'<text class="label" x="{x:.1f}" y="{height - 18}" text-anchor="middle">{tick:.2f}</text>')

    parts.append(f'<line class="axis" x1="{zero_x:.1f}" y1="{top - 10}" x2="{zero_x:.1f}" y2="{height - bottom + 10}"/>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        bar_y = y + 14
        value = float(row["co_bps"])
        x0 = min(zero_x, x_pos(value))
        bar_w = abs(x_pos(value) - zero_x)
        color = "#0f766e" if value >= 0 else "#b45309"
        parts.append(f'<text class="label" x="20" y="{y + 30}">{row["model_label"]}</text>')
        parts.append(f'<rect x="{x0:.1f}" y="{bar_y}" width="{max(bar_w, 1):.1f}" height="{bar_h}" rx="4" fill="{color}"/>')
        anchor = "start" if value >= 0 else "end"
        dx = 10 if value >= 0 else -10
        parts.append(
            f'<text class="value" x="{x_pos(value) + dx:.1f}" y="{bar_y + 17}" text-anchor="{anchor}">{value:.4f}</text>'
        )

    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def write_metric_panel_svg(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    width = 1680
    height = 430
    panel_gap = 48
    panel_width = (width - 80 - panel_gap) // 2
    left_margin = 24
    top = 78
    row_h = 48
    bar_h = 22
    bottom = 48
    max_rows = len(rows)
    chart_top = top + 38
    chart_height = row_h * max_rows

    def _panel(values_key: str, title: str, x_offset: int) -> list[str]:
        values = [float(row[values_key]) if row[values_key] is not None else 0.0 for row in rows]
        min_val = min(0.0, min(values))
        max_val = max(values)
        span = max_val - min_val
        pad = span * 0.1 if span > 0 else 0.1
        min_plot = min_val - pad
        max_plot = max_val + pad
        plot_left = x_offset + 260
        plot_right = x_offset + panel_width - 24
        plot_width = plot_right - plot_left

        def x_pos(value: float) -> float:
            return plot_left + ((value - min_plot) / (max_plot - min_plot)) * plot_width

        zero_x = x_pos(0.0)
        parts: list[str] = [
            f'<text class="panel-title" x="{x_offset + left_margin}" y="{top - 18}">{title}</text>',
        ]
        for tick in [min_plot, 0.0, max_plot]:
            x = x_pos(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{chart_top - 18}" x2="{x:.1f}" y2="{chart_top + chart_height - 4}"/>'
            )
            parts.append(
                f'<text class="tick" x="{x:.1f}" y="{height - 14}" text-anchor="middle">{tick:.2f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{zero_x:.1f}" y1="{chart_top - 18}" x2="{zero_x:.1f}" y2="{chart_top + chart_height - 4}"/>'
        )
        for idx, row in enumerate(rows):
            y = chart_top + idx * row_h
            value = float(row[values_key]) if row[values_key] is not None else 0.0
            x0 = min(zero_x, x_pos(value))
            bar_w = abs(x_pos(value) - zero_x)
            color = "#0f766e" if value >= 0 else "#b45309"
            parts.append(
                f'<text class="label" x="{x_offset + left_margin}" y="{y + 18}">{row["model_label"]}</text>'
            )
            parts.append(
                f'<rect x="{x0:.1f}" y="{y + 6}" width="{max(bar_w, 1):.1f}" height="{bar_h}" rx="4" fill="{color}"/>'
            )
            anchor = "start" if value >= 0 else "end"
            dx = 10 if value >= 0 else -10
            parts.append(
                f'<text class="value" x="{x_pos(value) + dx:.1f}" y="{y + 21}" text-anchor="{anchor}">{value:.4f}</text>'
            )
        return parts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, Helvetica, sans-serif; fill: #1b1f24; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.panel-title { font-size: 14px; font-weight: 700; }',
        '.label { font-size: 12px; }',
        '.tick { font-size: 11px; fill: #374151; }',
        '.value { font-size: 12px; font-weight: 600; }',
        '.axis { stroke: #9aa4af; stroke-width: 1; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="24" y="34">mc_maze model diagnostics</text>',
    ]
    parts.extend(_panel("co_bps", "co-bps", 0))
    parts.extend(_panel("vel_r2", "vel R2", panel_width + panel_gap))
    parts.append("</svg>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")
