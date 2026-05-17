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
from nlb_project.model_registry import get_spec
from nlb_project.pipeline import iter_cv_candidates


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


def test_explicit_candidates_load_and_replace_grid_axes(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"] = {
        "cv_folds": 2,
        "candidates": [{"n_components": 8}, {"n_components": 12, "ridge_alpha": 1.0}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(path)
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert [params["n_components"] for params, _label in candidates] == [8, 12]
    assert [params["ridge_alpha"] for params, _label in candidates] == [0.1, 1.0]


def test_explicit_candidates_reject_unknown_param(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"] = {"candidates": [{"not_a_param": 1}]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_explicit_candidates_reject_mixed_grid_keys(tmp_path: Path) -> None:
    raw = _valid_static_pca()
    raw["improvement"] = {
        "n_components_grid": [10],
        "ridge_alpha_grid": [0.1],
        "candidates": [{"n_components": 8}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be mixed"):
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
    assert cfg.baseline["lr_schedule"] == "constant"
    assert cfg.baseline["neuron_embedding_scale"] == 0.0
    assert cfg.baseline["ensemble_size"] == 1
    assert cfg.improvement["d_model_grid"] == [64]
    assert cfg.improvement["n_layers_grid"] == [2]
    assert cfg.improvement["dropout_grid"] == [0.1]
    assert cfg.improvement["lr_schedule_grid"] == ["constant"]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]
    assert cfg.improvement["ensemble_size"] == 1


def test_ndt_lite_ensemble_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_ensemble.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["ensemble_size"] == 1
    assert cfg.improvement["ensemble_size"] == 3
    assert cfg.improvement["lr_schedule_grid"] == ["constant"]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]


def test_ndt_lite_width_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_width_sweep.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.improvement["d_model_grid"] == [64, 128]
    assert cfg.improvement["n_layers_grid"] == [2]
    assert cfg.improvement["dropout_grid"] == [0.1]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]
    assert cfg.improvement["ensemble_size"] == 1


def test_ndt_lite_width_ensemble_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_width_ensemble.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.improvement["d_model_grid"] == [128]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]
    assert cfg.improvement["ensemble_size"] == 3


def test_ndt_lite_ensemble_size_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_ensemble_size_sweep.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 128
    assert cfg.baseline["ensemble_size"] == 3
    assert cfg.improvement["d_model_grid"] == [128]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]
    assert cfg.improvement["ensemble_size_grid"] == [3, 5]


def test_ndt_lite_arch_screen_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_arch_screen.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 128
    assert cfg.improvement["ensemble_size"] == 1
    assert len(candidates) == 11
    assert candidates[0][0]["d_model"] == 128
    assert candidates[-1][0]["d_model"] == 256
    assert all(params["ensemble_size"] == 1 for params, _label in candidates)


def test_ndt_lite_arch_5seed_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_arch_5seed_sweep.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["ensemble_size"] == 5
    assert len(candidates) == 2
    assert candidates[0][0]["d_model"] == 128
    assert candidates[1][0]["d_model"] == 192
    assert all(params["ensemble_size"] == 5 for params, _label in candidates)


def test_ndt_lite_192_ensemble_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_192_ensemble_sweep.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 192
    assert cfg.improvement["ensemble_size_grid"] == [5, 7]
    assert [params["ensemble_size"] for params, _label in candidates] == [5, 7]
    assert all(params["d_model"] == 192 for params, _label in candidates)


def test_ndt_lite_192_stability_screen_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_192_stability_screen.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 192
    assert cfg.improvement["ensemble_size"] == 1
    assert len(candidates) == 15
    assert candidates[0][0]["batch_size"] == 64
    assert candidates[10][0]["batch_size"] == 32
    assert all(params["ensemble_size"] == 1 for params, _label in candidates)


def test_ndt_lite_192_stability_5seed_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_192_stability_5seed_sweep.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["ensemble_size"] == 5
    assert len(candidates) == 2
    assert candidates[0][0]["heldin_loss_weight"] == 0.2
    assert candidates[1][0]["heldin_loss_weight"] == 0.3
    assert candidates[1][0]["validation_fraction"] == 0.05
    assert candidates[1][0]["max_epochs"] == 60
    assert all(params["ensemble_size"] == 5 for params, _label in candidates)


def test_ndt_lite_192_stability_ensemble_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_192_stability_ensemble_sweep.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["heldin_loss_weight"] == 0.3
    assert cfg.improvement["ensemble_size_grid"] == [5, 7]
    assert [params["ensemble_size"] for params, _label in candidates] == [5, 7]
    assert all(params["validation_fraction"] == 0.05 for params, _label in candidates)


def test_ndt_lite_tuning_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_tuning_sweep.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 128
    assert cfg.improvement["n_layers_grid"] == [2, 3]
    assert cfg.improvement["dropout_grid"] == [0.0, 0.1]
    assert cfg.improvement["mask_prob_grid"] == [0.1, 0.2]
    assert cfg.improvement["lr_schedule_grid"] == ["constant", "cosine"]
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0]


def test_ndt_lite_neuron_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_lite_neuron_sweep.yaml")

    assert cfg.model_type == "ndt_lite"
    assert cfg.baseline["d_model"] == 128
    assert cfg.baseline["neuron_embedding_scale"] == 0.0
    assert cfg.improvement["neuron_embedding_scale_grid"] == [0.0, 0.1, 0.25, 0.5]


def test_ndt_factorized_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_ndt_factorized_sweep.yaml")

    assert cfg.model_type == "ndt_factorized"
    assert cfg.baseline["d_model"] == 64
    assert cfg.baseline["n_layers"] == 2
    assert cfg.baseline["n_latents"] == 4
    assert cfg.improvement["d_model_grid"] == [64]
    assert cfg.improvement["n_layers_grid"] == [2]
    assert cfg.improvement["n_latents_grid"] == [4]
    assert cfg.improvement["ensemble_size"] == 1


def test_stndt_lite_screen_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_stndt_lite_screen.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "stndt_lite"
    assert cfg.baseline["d_model"] == 192
    assert cfg.baseline["spatial_n_heads"] == 4
    assert cfg.baseline["contrast_loss_weight"] == 0.0
    assert len(candidates) == 5
    assert candidates[1][0]["d_model"] == 256
    assert candidates[2][0]["n_layers"] == 3
    assert candidates[-1][0]["contrast_loss_weight"] == 0.03
    assert all(params["ensemble_size"] == 1 for params, _label in candidates)


def test_stndt_lite_ensemble_sweep_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_stndt_lite_ensemble_sweep.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "stndt_lite"
    assert cfg.baseline["n_layers"] == 3
    assert cfg.baseline["ensemble_size"] == 1
    assert [params["ensemble_size"] for params, _label in candidates] == [1, 5]
    assert all(params["contrast_loss_weight"] == 0.0 for params, _label in candidates)


def test_stndt_axial_screen_config_loads_without_torch() -> None:
    cfg = load_config("configs/benchmarks/mc_maze_stndt_axial_screen.yaml")
    candidates = list(iter_cv_candidates(get_spec(cfg.model_type), cfg))

    assert cfg.model_type == "stndt_axial"
    assert cfg.baseline["d_model"] == 64
    assert cfg.baseline["batch_size"] == 32
    assert cfg.baseline["spatial_n_heads"] == 4
    assert cfg.baseline["n_spatial_latents"] == 16
    assert len(candidates) == 1
    assert candidates[0][0]["n_layers"] == 1
    assert candidates[0][0]["n_spatial_latents"] == 16
    assert all(params["ensemble_size"] == 1 for params, _label in candidates)
