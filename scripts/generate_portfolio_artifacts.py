from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from nlb_project.reporting import (
        build_comparison_rows,
        write_comparison_csv,
        write_comparison_md,
        write_metric_panel_svg,
        write_comparison_svg,
        write_experiment_log_md,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nlb_project.reporting import (  # noqa: E402
        build_comparison_rows,
        write_comparison_csv,
        write_comparison_md,
        write_metric_panel_svg,
        write_comparison_svg,
        write_experiment_log_md,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio-facing benchmark comparison artifacts")
    parser.add_argument("--root", default=".", help="Repo root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "results" / "benchmark_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_comparison_rows(root)
    write_comparison_csv(rows, out_dir / "model_comparison.csv")
    write_comparison_md(rows, out_dir / "model_comparison.md")
    write_comparison_svg(rows, out_dir / "model_comparison.svg")
    write_metric_panel_svg(rows, out_dir / "model_diagnostics.svg")
    write_experiment_log_md(rows, root / "results" / "mc_maze" / "metrics.csv", out_dir / "experiment_log.md")


if __name__ == "__main__":
    main()
