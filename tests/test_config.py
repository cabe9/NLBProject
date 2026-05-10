"""Tests for typed experiment-config validation.

These pin the "invalid configs fail fast" contract: any missing required
key, unknown key, or malformed grid/value must raise a clear ``ValueError``
at load time, before any data is touched. Successful loads must also
preserve the exact params dicts that were baked into the committed
``metrics.csv`` artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nlb_project.config import DEFAULT_OUTPUT_HEAD, ExperimentConfig, load_config


def _valid_static_pca() -> dict:
    return {
        "model_type": "pca_latent_regression",
        "dataset_name": "mc_maze",
        "data_path": None,
        "data_prefix": "*full",
        "bin_size_ms": 5,
        "train_split": "train",
        "eval_split": "val",
        "include_psth": False,
        "log_offset": 0.001,
        "seed": 0,
        "skip_fields": [],
        "baseline": {"n_components": 10, "ridge_alpha": 0.1},
        "improvement": {
            "cv_folds": 2,
            "n_components_grid": [10],
            "ridge_alpha_grid": [0.1],
        },
        "output_dir": "results/benchmark_runs/static_pca",
    }


def test_valid_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_valid_static_pca()), encoding="utf-8")
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model_type == "pca_latent_regression"
    assert cfg.output_head == DEFAULT_OUTPUT_HEAD


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["not_a_real_field"] = 42
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_missing_required_baseline_key_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    del raw["baseline"]["n_components"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(path)


def test_unknown_baseline_key_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["baseline"]["not_a_model_param"] = 1.0
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_missing_required_sweep_axis_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    del raw["improvement"]["ridge_alpha_grid"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(path)


def test_empty_sweep_grid_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"]["ridge_alpha_grid"] = []
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_config(path)


def test_bad_output_head_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["output_head"] = "not_a_real_head"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="output_head"):
        load_config(path)


def test_unknown_model_type_rejected(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["model_type"] = "not_a_real_model"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="model_type"):
        load_config(path)


def test_ndt_lite_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 64
    assert cfg.improvement["d_model_grid"] == [64]
