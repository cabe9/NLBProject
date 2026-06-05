"""Tests for typed experiment-config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nlb_project.config import DEFAULT_OUTPUT_HEAD, ExperimentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_select_on_full_val_flag_requires_candidates(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"]["select_on_full_val_all_candidates"] = True
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="select_on_full_val_all_candidates requires"):
        load_config(path)


def test_select_on_full_val_flag_accepts_explicit_candidates(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"] = {
        "select_on_full_val_all_candidates": True,
        "candidates": [
            {"n_components": 10, "ridge_alpha": 0.1},
            {"n_components": 12, "ridge_alpha": 0.2},
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.improvement["select_on_full_val_all_candidates"] is True


def test_select_on_full_val_flag_must_be_bool(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"] = {
        "select_on_full_val_all_candidates": 1,
        "candidates": [{"n_components": 10, "ridge_alpha": 0.1}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="expected bool"):
        load_config(path)


@pytest.mark.parametrize(
    "config_name",
    [
        "mc_maze_stndt_lite_identity_objective_5seed.yaml",
        "mc_maze_stndt_lite_cd_reconcile_screen.yaml",
        "mc_maze_stndt_lite_screen_q_block_mask.yaml",
        "mc_maze_stndt_lite_screen_r1_unit_calibration.yaml",
    ],
)
def test_benchmark_configs_load(config_name: str) -> None:
    pytest.importorskip("nlb_project.model_registry")
    from nlb_project.model_registry import MODEL_REGISTRY

    if "stndt_lite" not in MODEL_REGISTRY:
        pytest.skip("stndt_lite not registered (restore model_registry.py)")

    path = REPO_ROOT / "configs" / "benchmarks" / config_name
    cfg = load_config(path)
    assert cfg.model_type == "stndt_lite"
    if config_name.endswith("cd_reconcile_screen.yaml"):
        assert cfg.improvement.get("select_on_full_val_all_candidates") is True
        assert len(cfg.improvement["candidates"]) == 7
    if config_name.endswith("screen_q_block_mask.yaml"):
        assert cfg.improvement.get("select_on_full_val_all_candidates") is True
        assert len(cfg.improvement["candidates"]) == 4
    if config_name.endswith("screen_r1_unit_calibration.yaml"):
        assert cfg.improvement.get("select_on_full_val_all_candidates") is True
        assert len(cfg.improvement["candidates"]) == 4
