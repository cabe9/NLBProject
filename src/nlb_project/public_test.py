"""Public-test evaluation helpers for frozen NLB'21 leaderboards.

EvalAI submissions closed in January 2026, but the NLB maintainers made the
test-split evaluation HDF5 public in the ``neurallatents/nlb_tools`` repo.
This module keeps test evaluation separate from the train/val experiment
pipeline so we never accidentally treat ``test`` like a validation split.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

from .config import ExperimentConfig
from .data_contract import resolve_data_path
from .io_utils import ensure_dir, write_metrics_csv
from .model_registry import ModelSpec, get_spec
from .pipeline import (
    _dataset_key,
    _select_best_params,
    _split_key,
    build_reference_params,
    set_seeds,
)
from .run_metadata import build_run_metadata

logger = logging.getLogger(__name__)

PUBLIC_TEST_EVAL_DATA_URL = (
    "https://media.githubusercontent.com/media/neurallatents/nlb_tools/main/data/eval_data_test.h5"
)
PUBLIC_TEST_EVAL_DATA_SHA256 = "f2f434d30193251b58e6a693381fdd185c8fbc0e47d38f8810e65295bc42c865"
PUBLIC_TEST_EVAL_DATA_SIZE_BYTES = 105_927_616


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    size_bytes: int
    downloaded: bool


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_public_test_eval_data(
    out_path: str | Path,
    *,
    force: bool = False,
    url: str = PUBLIC_TEST_EVAL_DATA_URL,
    expected_sha256: str = PUBLIC_TEST_EVAL_DATA_SHA256,
) -> DownloadResult:
    """Download the official public NLB test target HDF5 if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        actual_sha = sha256_file(out_path)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"{out_path} exists but has sha256 {actual_sha}; "
                f"expected {expected_sha256}. Pass force=True to replace it."
            )
        return DownloadResult(
            path=out_path,
            sha256=actual_sha,
            size_bytes=out_path.stat().st_size,
            downloaded=False,
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    tmp.write(chunk)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    assert tmp_path is not None
    actual_sha = sha256_file(tmp_path)
    if actual_sha != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded public test target has sha256 {actual_sha}; expected {expected_sha256}."
        )

    tmp_path.replace(out_path)
    return DownloadResult(
        path=out_path,
        sha256=actual_sha,
        size_bytes=out_path.stat().st_size,
        downloaded=True,
    )


def _metrics_for_split(
    eval_data_path: str | Path,
    output_dict: dict[str, Any],
    *,
    dataset_name: str,
    bin_size_ms: int,
) -> dict[str, Any]:
    split_key = _split_key(dataset_name, bin_size_ms)
    results = evaluate(str(eval_data_path), output_dict)
    for result in results:
        if split_key in result:
            return result[split_key]
    available = sorted(key for result in results for key in result)
    raise KeyError(f"Evaluation target did not return `{split_key}`; available splits: {available}")


def _run_public_test_fit(
    dataset: NWBDataset,
    cfg: ExperimentConfig,
    spec: ModelSpec,
    params: dict[str, Any],
    *,
    eval_data_path: str | Path,
    train_trial_split: list[str],
    run_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on the configured training split and score predictions on test."""
    logger.info("[%s public-test] model_type=%s params=%s", run_name, spec.name, params)
    train_dict = make_train_input_tensors(
        dataset, cfg.dataset_name, trial_split=train_trial_split, save_file=False
    )
    eval_dict = make_eval_input_tensors(
        dataset, cfg.dataset_name, trial_split="test", save_file=False
    )

    extra_kwargs = spec.extra_predict_kwargs_fn(cfg)
    preds = spec.predict(
        train_dict["train_spikes_heldin"],
        train_dict["train_spikes_heldout"],
        eval_dict["eval_spikes_heldin"],
        **params,
        **extra_kwargs,
    )

    output_dict = {_dataset_key(cfg.dataset_name, cfg.bin_size_ms): preds}
    metrics = _metrics_for_split(
        eval_data_path,
        output_dict,
        dataset_name=cfg.dataset_name,
        bin_size_ms=cfg.bin_size_ms,
    )
    return output_dict, metrics


def _write_public_summary(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    lines = [
        "# NLB Public Test Summary",
        "",
        "Evaluated locally against the public NLB test target HDF5.",
        "",
        "| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {co_bps} | {vel_r2} | {psth_r2} | {fp_bps} | {params} |".format(
                model=row["model"],
                split=row["split"],
                co_bps=fmt(row.get("co-bps")),
                vel_r2=fmt(row.get("vel R2")),
                psth_r2=fmt(row.get("psth R2")),
                fp_bps=fmt(row.get("fp-bps")),
                params=row["params"],
            )
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def run_public_test_evaluation(
    cfg: ExperimentConfig,
    *,
    eval_data_path: str | Path,
    config_path: str | None = None,
    output_dir: str | Path | None = None,
    final_train_trial_split: list[str] | None = None,
) -> dict[str, Any]:
    """Run reference and CV-selected configs on the NLB public test split."""
    set_seeds(cfg.seed)
    final_train_trial_split = final_train_trial_split or ["train", "val"]
    eval_data_path = Path(eval_data_path)
    if not eval_data_path.exists():
        raise FileNotFoundError(
            f"Public test eval data not found at {eval_data_path}. "
            "Run `nlb-get-public-eval-data` first."
        )

    out_dir = ensure_dir(output_dir or Path("results") / "public_test" / cfg.dataset_name)
    pred_dir = ensure_dir(out_dir / "predictions")

    spec = get_spec(cfg.model_type)
    dataset_path = resolve_data_path(cfg.dataset_name, cfg.data_path, cfg.data_prefix)
    dataset = NWBDataset(dataset_path, cfg.data_prefix, skip_fields=cfg.skip_fields)
    dataset.resample(cfg.bin_size_ms)

    reference_params = build_reference_params(spec, cfg)
    selected_params = _select_best_params(dataset, cfg, spec)

    reference_output, reference_metrics = _run_public_test_fit(
        dataset,
        cfg,
        spec,
        reference_params,
        eval_data_path=eval_data_path,
        train_trial_split=final_train_trial_split,
        run_name="reference",
    )
    reference_path = pred_dir / "baseline_public_test_predictions.h5"
    save_to_h5(reference_output, str(reference_path), overwrite=True)

    selected_output, selected_metrics = _run_public_test_fit(
        dataset,
        cfg,
        spec,
        selected_params,
        eval_data_path=eval_data_path,
        train_trial_split=final_train_trial_split,
        run_name="selected",
    )
    selected_path = pred_dir / "selected_public_test_predictions.h5"
    save_to_h5(selected_output, str(selected_path), overwrite=True)

    split_key = _split_key(cfg.dataset_name, cfg.bin_size_ms)
    final_train_label = "+".join(final_train_trial_split)
    rows = [
        {
            "model": "baseline",
            "model_type": cfg.model_type,
            "split": split_key,
            "train_split": final_train_label,
            "model_selection_train_split": cfg.train_split,
            "model_selection_eval_split": cfg.eval_split,
            "eval_split": "test",
            "co-bps": reference_metrics.get("co-bps"),
            "vel R2": reference_metrics.get("vel R2"),
            "psth R2": reference_metrics.get("psth R2"),
            "fp-bps": reference_metrics.get("fp-bps"),
            "params": json.dumps(reference_params, sort_keys=True),
        },
        {
            "model": "selected",
            "model_type": cfg.model_type,
            "split": split_key,
            "train_split": final_train_label,
            "model_selection_train_split": cfg.train_split,
            "model_selection_eval_split": cfg.eval_split,
            "eval_split": "test",
            "co-bps": selected_metrics.get("co-bps"),
            "vel R2": selected_metrics.get("vel R2"),
            "psth R2": selected_metrics.get("psth R2"),
            "fp-bps": selected_metrics.get("fp-bps"),
            "params": json.dumps(selected_params, sort_keys=True),
        },
    ]
    write_metrics_csv(rows, out_dir / "metrics.csv")
    _write_public_summary(rows, out_dir / "summary.md")

    metadata = build_run_metadata(
        cfg,
        config_path=config_path,
        dataset_path=dataset_path,
        output_dir=out_dir,
        baseline_metrics=reference_metrics,
        improved_metrics=selected_metrics,
        baseline_params=reference_params,
        improved_params=selected_params,
        prediction_artifacts={
            "baseline_public_test_predictions": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "selected_public_test_predictions": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
            },
        },
    )
    metadata["public_test_eval_data"] = {
        "path": str(eval_data_path),
        "sha256": sha256_file(eval_data_path),
        "source_url": PUBLIC_TEST_EVAL_DATA_URL,
    }
    metadata["data"]["eval_split"] = "test"
    metadata["data"]["train_split"] = final_train_label
    metadata["data"]["model_selection_train_split"] = cfg.train_split
    metadata["data"]["model_selection_eval_split"] = cfg.eval_split
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {
        "baseline_metrics": reference_metrics,
        "selected_metrics": selected_metrics,
        "baseline_params": reference_params,
        "selected_params": selected_params,
        "output_dir": out_dir,
        "prediction_paths": {
            "baseline": reference_path,
            "selected": selected_path,
        },
    }


def read_public_metrics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
