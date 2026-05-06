from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path

from nlb_project.reporting import ComparisonSpec, build_comparison_rows, write_comparison_csv
from nlb_project.result_provenance import (
    validate_comparison_artifacts,
    validate_run_directory,
)


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "model_type", "co-bps", "vel R2", "psth R2", "params"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_valid_run(root: Path, run_name: str = "demo") -> Path:
    run_dir = root / "results" / "benchmark_runs" / run_name
    baseline_path = run_dir / "predictions" / "baseline_predictions.h5"
    improved_path = run_dir / "predictions" / "improved_predictions.h5"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(b"baseline")
    improved_path.write_bytes(b"improved")

    baseline_params = {"ridge_alpha": 0.1}
    improved_params = {"ridge_alpha": 1.0}
    _write_metrics(
        run_dir / "metrics.csv",
        [
            {
                "model": "baseline",
                "model_type": "ridge_direct",
                "co-bps": "0.01",
                "vel R2": "0.2",
                "psth R2": "",
                "params": json.dumps(baseline_params, sort_keys=True),
            },
            {
                "model": "improved",
                "model_type": "ridge_direct",
                "co-bps": "0.03",
                "vel R2": "0.4",
                "psth R2": "",
                "params": json.dumps(improved_params, sort_keys=True),
            },
        ],
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "metadata_schema_version": 1,
                "config": {
                    "model_type": "ridge_direct",
                    "output_dir": f"results/benchmark_runs/{run_name}",
                },
                "artifacts": {
                    "predictions": {
                        "baseline_predictions": {
                            "path": f"results/benchmark_runs/{run_name}/predictions/baseline_predictions.h5",
                            "sha256": sha256(baseline_path.read_bytes()).hexdigest(),
                        },
                        "improved_predictions": {
                            "path": f"results/benchmark_runs/{run_name}/predictions/improved_predictions.h5",
                            "sha256": sha256(improved_path.read_bytes()).hexdigest(),
                        },
                    }
                },
                "baseline_metrics": {"co-bps": 0.01, "vel R2": 0.2, "psth R2": None},
                "improved_metrics": {"co-bps": 0.03, "vel R2": 0.4, "psth R2": None},
                "baseline_params": baseline_params,
                "improved_params": improved_params,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_validate_run_directory_accepts_consistent_metadata(tmp_path: Path) -> None:
    run_dir = _write_valid_run(tmp_path)

    report = validate_run_directory(tmp_path, run_dir)

    assert report.ok
    assert report.checked_runs == 1
    assert report.warnings == []


def test_validate_run_directory_rejects_stale_params(tmp_path: Path) -> None:
    run_dir = _write_valid_run(tmp_path)
    _write_metrics(
        run_dir / "metrics.csv",
        [
            {
                "model": "baseline",
                "model_type": "ridge_direct",
                "co-bps": "0.01",
                "vel R2": "0.2",
                "psth R2": "",
                "params": json.dumps({"ridge_alpha": 999.0}, sort_keys=True),
            },
            {
                "model": "improved",
                "model_type": "ridge_direct",
                "co-bps": "0.03",
                "vel R2": "0.4",
                "psth R2": "",
                "params": json.dumps({"ridge_alpha": 1.0}, sort_keys=True),
            },
        ],
    )

    report = validate_run_directory(tmp_path, run_dir)

    assert not report.ok
    assert any("baseline.params" in issue.message for issue in report.errors)


def test_validate_run_directory_warns_when_local_metadata_is_absent(tmp_path: Path) -> None:
    run_dir = _write_valid_run(tmp_path)
    (run_dir / "run_metadata.json").unlink()

    report = validate_run_directory(tmp_path, run_dir)

    assert report.ok
    assert report.checked_runs == 1
    assert any("metadata checks skipped" in issue.message for issue in report.warnings)


def test_validate_comparison_artifacts_rejects_stale_csv(tmp_path: Path) -> None:
    run_dir = _write_valid_run(tmp_path)
    specs = [
        ComparisonSpec(
            label="demo ridge",
            metrics_path=f"{run_dir.relative_to(tmp_path).as_posix()}/metrics.csv",
            model_row="improved",
            note="fixture row",
        )
    ]
    comparison_path = tmp_path / "results" / "benchmark_runs" / "model_comparison.csv"
    write_comparison_csv(build_comparison_rows(tmp_path, specs), comparison_path)

    assert validate_comparison_artifacts(tmp_path, specs).ok

    text = comparison_path.read_text(encoding="utf-8")
    comparison_path.write_text(text.replace("0.03", "0.99"), encoding="utf-8")

    report = validate_comparison_artifacts(tmp_path, specs)

    assert not report.ok
    assert any("co_bps" in issue.message for issue in report.errors)
