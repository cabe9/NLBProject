from __future__ import annotations

from pathlib import Path

from nlb_project.config import ExperimentConfig
from nlb_project.run_metadata import build_run_metadata


def _cfg(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_type="pca_latent_regression",
        dataset_name="mc_maze",
        data_path=None,
        data_prefix="*full",
        bin_size_ms=5,
        train_split="train",
        eval_split="val",
        include_psth=False,
        log_offset=0.001,
        seed=0,
        skip_fields=[],
        baseline={"n_components": 10, "ridge_alpha": 0.1},
        improvement={"n_components_grid": [10], "ridge_alpha_grid": [0.1]},
        output_dir=str(output_dir),
        output_head="log_link",
    )


def test_build_run_metadata_records_audit_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "nlb_project.run_metadata.collect_runtime_metadata",
        lambda: {"python": {"version": "test-python"}},
    )
    monkeypatch.setattr(
        "nlb_project.run_metadata.collect_git_metadata",
        lambda: {"available": True, "commit": "abc123", "dirty": False},
    )

    metadata = build_run_metadata(
        _cfg(tmp_path / "run"),
        config_path="configs/example.yaml",
        dataset_path=tmp_path / "data",
        output_dir=tmp_path / "run",
        baseline_metrics={"co-bps": 0.01},
        improved_metrics={"co-bps": 0.02},
        baseline_params={"n_components": 10},
        improved_params={"n_components": 20},
        prediction_artifacts={
            "baseline_predictions": {
                "path": "baseline_predictions.h5",
                "sha256": "baseline-sha",
            }
        },
    )

    assert metadata["metadata_schema_version"] == 1
    assert metadata["config_path"] == "configs/example.yaml"
    assert metadata["config"]["model_type"] == "pca_latent_regression"
    assert metadata["runtime"]["python"]["version"] == "test-python"
    assert metadata["git"]["commit"] == "abc123"
    assert metadata["data"]["resolved_data_path"] == str(tmp_path / "data")
    assert metadata["artifacts"]["predictions"]["baseline_predictions"]["sha256"] == (
        "baseline-sha"
    )
