"""Run metadata helpers for reproducible experiment artifacts."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGE_DISTRIBUTIONS = (
    "nlb-project",
    "nlb-tools",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "h5py",
    "PyYAML",
)


def _stringify_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _run_git_command(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_git_metadata() -> dict[str, Any]:
    """Return best-effort git context for the current working tree."""
    root = _run_git_command(["rev-parse", "--show-toplevel"])
    if root is None:
        return {"available": False}

    status_short = _run_git_command(["status", "--short"]) or ""
    return {
        "available": True,
        "root": root,
        "branch": _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "remote_origin": _run_git_command(["config", "--get", "remote.origin.url"]),
        "dirty": bool(status_short),
        "status_short": status_short.splitlines(),
    }


def collect_package_versions(
    distributions: Sequence[str] = _PACKAGE_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions for packages that materially affect a run."""
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def collect_runtime_metadata() -> dict[str, Any]:
    """Return Python, platform, package, and command metadata."""
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": collect_package_versions(),
        "command": sys.argv,
    }


def build_run_metadata(
    cfg: Any,
    *,
    config_path: str | Path | None,
    dataset_path: str | Path,
    output_dir: str | Path,
    baseline_metrics: dict[str, Any],
    improved_metrics: dict[str, Any],
    baseline_params: dict[str, Any],
    improved_params: dict[str, Any],
    prediction_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-serializable metadata payload for an experiment run."""
    return {
        "metadata_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "config_path": _stringify_path(config_path),
        "config": asdict(cfg),
        "runtime": collect_runtime_metadata(),
        "git": collect_git_metadata(),
        "data": {
            "dataset_name": cfg.dataset_name,
            "resolved_data_path": _stringify_path(dataset_path),
            "data_prefix": cfg.data_prefix,
            "bin_size_ms": cfg.bin_size_ms,
            "train_split": cfg.train_split,
            "eval_split": cfg.eval_split,
        },
        "artifacts": {
            "output_dir": _stringify_path(output_dir),
            "predictions": dict(prediction_artifacts),
        },
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "baseline_params": baseline_params,
        "improved_params": improved_params,
    }


__all__ = [
    "build_run_metadata",
    "collect_git_metadata",
    "collect_package_versions",
    "collect_runtime_metadata",
]
