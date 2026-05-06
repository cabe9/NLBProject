"""Validation helpers for committed benchmark result artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .reporting import DEFAULT_COMPARISON_SPECS, ComparisonSpec, build_comparison_rows

_METRIC_COLUMNS = ("co-bps", "vel R2", "psth R2")
_COMPARISON_FIELDS = (
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
)


@dataclass(frozen=True)
class ProvenanceIssue:
    path: str
    message: str


@dataclass
class ProvenanceReport:
    checked_runs: int
    errors: list[ProvenanceIssue]
    warnings: list[ProvenanceIssue]

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_missing(value: Any) -> Any:
    if value == "":
        return None
    return value


def _metric_values_match(csv_value: str | None, metadata_value: Any) -> bool:
    csv_value = _normalize_missing(csv_value)
    if csv_value is None and metadata_value is None:
        return True
    if csv_value is None or metadata_value is None:
        return False
    try:
        return abs(float(csv_value) - float(metadata_value)) < 1e-12
    except (TypeError, ValueError):
        return str(csv_value) == str(metadata_value)


def _normalize_comparison_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _prediction_artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "baseline_predictions": run_dir / "predictions" / "baseline_predictions.h5",
        "improved_predictions": run_dir / "predictions" / "improved_predictions.h5",
    }


def _resolve_recorded_path(root: Path, recorded_path: Any) -> Path | None:
    if not isinstance(recorded_path, str) or not recorded_path:
        return None
    path = Path(recorded_path)
    if path.is_absolute():
        return path
    return root / path


def validate_run_directory(root: str | Path, run_dir: str | Path) -> ProvenanceReport:
    """Validate one result directory's metrics, metadata, and prediction files."""
    root = Path(root)
    run_dir = Path(run_dir)
    errors: list[ProvenanceIssue] = []
    warnings: list[ProvenanceIssue] = []

    metrics_path = run_dir / "metrics.csv"
    metadata_path = run_dir / "run_metadata.json"

    if not metrics_path.exists():
        errors.append(ProvenanceIssue(_display_path(root, metrics_path), "missing metrics.csv"))
        return ProvenanceReport(checked_runs=0, errors=errors, warnings=warnings)

    try:
        rows = _read_csv_rows(metrics_path)
    except (csv.Error, OSError) as exc:
        errors.append(
            ProvenanceIssue(_display_path(root, metrics_path), f"could not read CSV: {exc}")
        )
        return ProvenanceReport(checked_runs=0, errors=errors, warnings=warnings)

    row_by_model = {row.get("model"): row for row in rows}
    for model_row in ("baseline", "improved"):
        if model_row not in row_by_model:
            errors.append(
                ProvenanceIssue(_display_path(root, metrics_path), f"missing `{model_row}` row")
            )
            continue
        try:
            json.loads(row_by_model[model_row]["params"])
        except (KeyError, json.JSONDecodeError) as exc:
            errors.append(
                ProvenanceIssue(
                    _display_path(root, metrics_path),
                    f"{model_row}.params is not valid JSON: {exc}",
                )
            )

    if not metadata_path.exists():
        warnings.append(
            ProvenanceIssue(
                _display_path(root, metadata_path),
                "run_metadata.json is not present locally; metadata checks skipped",
            )
        )
        return ProvenanceReport(checked_runs=1, errors=errors, warnings=warnings)

    try:
        metadata = _load_json(metadata_path)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        errors.append(
            ProvenanceIssue(_display_path(root, metadata_path), f"could not read JSON: {exc}")
        )
        return ProvenanceReport(checked_runs=0, errors=errors, warnings=warnings)

    config = metadata.get("config", {})
    if not isinstance(config, dict):
        errors.append(
            ProvenanceIssue(_display_path(root, metadata_path), "`config` must be an object")
        )
        config = {}

    expected_output_dir = _display_path(root, run_dir)
    if config.get("output_dir") != expected_output_dir:
        errors.append(
            ProvenanceIssue(
                _display_path(root, metadata_path),
                f"config.output_dir={config.get('output_dir')!r} does not match {expected_output_dir!r}",
            )
        )

    for model_row, metadata_prefix in (("baseline", "baseline"), ("improved", "improved")):
        if model_row not in row_by_model:
            continue

        row = row_by_model[model_row]
        if row.get("model_type") != config.get("model_type"):
            errors.append(
                ProvenanceIssue(
                    _display_path(root, metrics_path),
                    f"{model_row}.model_type={row.get('model_type')!r} does not match metadata config.model_type={config.get('model_type')!r}",
                )
            )

        metadata_metrics = metadata.get(f"{metadata_prefix}_metrics", {})
        if not isinstance(metadata_metrics, dict):
            errors.append(
                ProvenanceIssue(
                    _display_path(root, metadata_path),
                    f"`{metadata_prefix}_metrics` must be an object",
                )
            )
            metadata_metrics = {}

        for metric_name in _METRIC_COLUMNS:
            if metric_name not in row:
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, metrics_path),
                        f"{model_row} row missing `{metric_name}` column",
                    )
                )
                continue
            if not _metric_values_match(row.get(metric_name), metadata_metrics.get(metric_name)):
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, metrics_path),
                        f"{model_row}.{metric_name}={row.get(metric_name)!r} does not match metadata value {metadata_metrics.get(metric_name)!r}",
                    )
                )

        try:
            csv_params = json.loads(row["params"])
        except (KeyError, json.JSONDecodeError) as exc:
            errors.append(
                ProvenanceIssue(
                    _display_path(root, metrics_path),
                    f"{model_row}.params is not valid JSON: {exc}",
                )
            )
            continue

        metadata_params = metadata.get(f"{metadata_prefix}_params")
        if csv_params != metadata_params:
            errors.append(
                ProvenanceIssue(
                    _display_path(root, metrics_path),
                    f"{model_row}.params does not match `{metadata_prefix}_params` in metadata",
                )
            )

    for name, path in _prediction_artifact_paths(run_dir).items():
        if not path.exists():
            warnings.append(
                ProvenanceIssue(
                    _display_path(root, path),
                    f"{name} artifact is not present locally; hash check may be skipped",
                )
            )

    artifacts_metadata = metadata.get("artifacts", {})
    prediction_metadata = (
        artifacts_metadata.get("predictions", {}) if isinstance(artifacts_metadata, dict) else {}
    )
    if prediction_metadata:
        for name, artifact in prediction_metadata.items():
            if not isinstance(artifact, dict):
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, metadata_path),
                        f"prediction artifact metadata for `{name}` must be an object",
                    )
                )
                continue
            artifact_path = artifact.get("path")
            artifact_sha = artifact.get("sha256")
            if not artifact_path or not artifact_sha:
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, metadata_path),
                        f"prediction artifact `{name}` must include path and sha256",
                    )
                )
                continue
            resolved_path = _resolve_recorded_path(root, artifact_path)
            if resolved_path is None or not resolved_path.exists():
                warnings.append(
                    ProvenanceIssue(
                        _display_path(root, metadata_path),
                        f"prediction artifact `{name}` is not present locally at {artifact_path!r}; sha256 check skipped",
                    )
                )
                continue
            actual_sha = sha256(resolved_path.read_bytes()).hexdigest()
            if actual_sha != artifact_sha:
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, resolved_path),
                        f"sha256 for prediction artifact `{name}` is {actual_sha}, expected {artifact_sha}",
                    )
                )
    else:
        warnings.append(
            ProvenanceIssue(
                _display_path(root, metadata_path),
                "legacy metadata has no prediction artifact hashes; rerun to capture schema v1 provenance",
            )
        )

    return ProvenanceReport(checked_runs=1, errors=errors, warnings=warnings)


def validate_comparison_artifacts(
    root: str | Path,
    specs: list[ComparisonSpec] | None = None,
) -> ProvenanceReport:
    """Validate generated comparison CSV against the reporting manifest."""
    root = Path(root)
    comparison_specs = specs or DEFAULT_COMPARISON_SPECS
    errors: list[ProvenanceIssue] = []
    warnings: list[ProvenanceIssue] = []
    comparison_path = root / "results" / "benchmark_runs" / "model_comparison.csv"

    if not comparison_path.exists():
        return ProvenanceReport(
            checked_runs=0,
            errors=[
                ProvenanceIssue(
                    _display_path(root, comparison_path), "missing model_comparison.csv"
                )
            ],
            warnings=[],
        )

    try:
        actual_rows = _read_csv_rows(comparison_path)
    except (csv.Error, OSError) as exc:
        return ProvenanceReport(
            checked_runs=0,
            errors=[
                ProvenanceIssue(
                    _display_path(root, comparison_path),
                    f"could not read comparison CSV: {exc}",
                )
            ],
            warnings=[],
        )

    try:
        expected_rows = build_comparison_rows(root, comparison_specs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        expected_rows = []
        errors.append(
            ProvenanceIssue(
                _display_path(root, comparison_path),
                f"could not rebuild comparison rows from manifest: {exc}",
            )
        )

    if len(actual_rows) != len(expected_rows):
        errors.append(
            ProvenanceIssue(
                _display_path(root, comparison_path),
                f"expected {len(expected_rows)} comparison rows, found {len(actual_rows)}",
            )
        )

    for idx, (actual, expected) in enumerate(zip(actual_rows, expected_rows, strict=False)):
        for field in _COMPARISON_FIELDS:
            actual_value = actual.get(field, "")
            expected_value = _normalize_comparison_value(expected.get(field, ""))
            if actual_value != expected_value:
                errors.append(
                    ProvenanceIssue(
                        _display_path(root, comparison_path),
                        f"row {idx + 1} field `{field}` is {actual_value!r}, expected {expected_value!r}",
                    )
                )

    for spec in comparison_specs:
        source_path = root / spec.metrics_path
        if not source_path.exists():
            errors.append(
                ProvenanceIssue(
                    _display_path(root, source_path),
                    "comparison manifest points at a missing metrics file",
                )
            )
            continue
        source_rows = _read_csv_rows(source_path)
        if spec.model_row not in {row.get("model") for row in source_rows}:
            errors.append(
                ProvenanceIssue(
                    _display_path(root, source_path),
                    f"comparison manifest selects missing row `{spec.model_row}`",
                )
            )

    return ProvenanceReport(checked_runs=0, errors=errors, warnings=warnings)


def validate_results(root: str | Path = ".") -> ProvenanceReport:
    """Validate committed benchmark and headline result artifacts."""
    root = Path(root)
    run_dirs = sorted((root / "results" / "benchmark_runs").glob("*/metrics.csv"))
    main_metrics = root / "results" / "mc_maze" / "metrics.csv"
    if main_metrics.exists():
        run_dirs.append(main_metrics)

    errors: list[ProvenanceIssue] = []
    warnings: list[ProvenanceIssue] = []
    checked_runs = 0

    for metrics_path in run_dirs:
        report = validate_run_directory(root, metrics_path.parent)
        checked_runs += report.checked_runs
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    comparison_report = validate_comparison_artifacts(root)
    errors.extend(comparison_report.errors)
    warnings.extend(comparison_report.warnings)

    return ProvenanceReport(
        checked_runs=checked_runs,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    "ProvenanceIssue",
    "ProvenanceReport",
    "validate_comparison_artifacts",
    "validate_results",
    "validate_run_directory",
]
