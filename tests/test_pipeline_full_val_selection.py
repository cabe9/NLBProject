"""Pipeline tests for full train/val candidate selection (Screen C)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from nlb_project.config import ExperimentConfig
from nlb_project.model_registry import ModelSpec
from nlb_project.pipeline import _select_best_params, _select_best_params_full_val

if "nlb_tools" not in sys.modules:
    try:
        __import__("nlb_tools")
    except ModuleNotFoundError:
        nlb_tools = types.ModuleType("nlb_tools")
        evaluation = types.ModuleType("nlb_tools.evaluation")
        make_tensors = types.ModuleType("nlb_tools.make_tensors")
        nwb_interface = types.ModuleType("nlb_tools.nwb_interface")

        evaluation.evaluate = lambda *_args, **_kwargs: [  # type: ignore[attr-defined]
            {"mc_maze_split": {"co-bps": 0.0, "vel R2": 0.0, "psth R2": None}}
        ]
        make_tensors.make_eval_input_tensors = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
        make_tensors.make_eval_target_tensors = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
        make_tensors.make_train_input_tensors = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
        make_tensors.save_to_h5 = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

        class _PlaceholderDataset:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def resample(self, _bin_size_ms: int) -> None:
                return None

        nwb_interface.NWBDataset = _PlaceholderDataset  # type: ignore[attr-defined]

        sys.modules["nlb_tools"] = nlb_tools
        sys.modules["nlb_tools.evaluation"] = evaluation
        sys.modules["nlb_tools.make_tensors"] = make_tensors
        sys.modules["nlb_tools.nwb_interface"] = nwb_interface


def _wire_pipeline_mocks(monkeypatch, *, co_bps_sequence: list[float]) -> ModelSpec:
    scores = iter(co_bps_sequence)

    class _FakeDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            self.trial_info = pd.DataFrame({"split": ["train", "train", "val", "val"]})

        def resample(self, _bin_size_ms: int) -> None:
            return None

    def _fake_train_tensors(_dataset, _dataset_name, trial_split, save_file=False):
        n = int(np.sum(trial_split)) if isinstance(trial_split, np.ndarray) else 2
        return {
            "train_spikes_heldin": np.ones((n, 5, 3), dtype=float),
            "train_spikes_heldout": np.ones((n, 5, 2), dtype=float),
        }

    def _fake_eval_tensors(_dataset, _dataset_name, trial_split, save_file=False):
        n = int(np.sum(trial_split)) if isinstance(trial_split, np.ndarray) else 2
        return {"eval_spikes_heldin": np.ones((n, 5, 3), dtype=float)}

    def _fake_target_tensors(
        _dataset,
        _dataset_name,
        train_trial_split,
        eval_trial_split,
        save_file=False,
        include_psth=False,
    ):
        n_eval = int(np.sum(eval_trial_split)) if isinstance(eval_trial_split, np.ndarray) else 2
        return {
            "mc_maze": {
                "eval_spikes_heldout": np.ones((n_eval, 5, 2), dtype=float),
                "train_behavior": np.zeros((2, 5, 2), dtype=float),
                "eval_behavior": np.zeros((n_eval, 5, 2), dtype=float),
            }
        }

    def _fake_evaluate(_target_dict, _output_dict):
        return [{"mc_maze_split": {"co-bps": float(next(scores)), "vel R2": 0.0, "psth R2": None}}]

    spec = ModelSpec(
        name="test_model",
        predict=lambda *_args, **_kwargs: {},
        baseline_params=(("ridge_alpha", float),),
        sweep_axes=(),
        uses_rate_head=False,
    )

    monkeypatch.setattr("nlb_project.pipeline.NWBDataset", _FakeDataset)
    monkeypatch.setattr("nlb_project.pipeline.make_train_input_tensors", _fake_train_tensors)
    monkeypatch.setattr("nlb_project.pipeline.make_eval_input_tensors", _fake_eval_tensors)
    monkeypatch.setattr("nlb_project.pipeline.make_eval_target_tensors", _fake_target_tensors)
    monkeypatch.setattr("nlb_project.pipeline.evaluate", _fake_evaluate)
    monkeypatch.setattr("nlb_project.pipeline.get_spec", lambda _model_type: spec)
    return spec


def test_select_best_params_full_val_evaluates_all_candidates(tmp_path: Path, monkeypatch) -> None:
    spec = _wire_pipeline_mocks(monkeypatch, co_bps_sequence=[0.10, 0.30, 0.20])

    cfg = ExperimentConfig(
        model_type="test_model",
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
        baseline={"ridge_alpha": 0.1},
        improvement={
            "select_on_full_val_all_candidates": True,
            "candidates": [
                {"ridge_alpha": 0.1},
                {"ridge_alpha": 0.2},
                {"ridge_alpha": 0.3},
            ],
        },
        output_dir=str(tmp_path / "run"),
    )

    class _FakeDataset:
        def __init__(self) -> None:
            self.trial_info = pd.DataFrame({"split": ["train", "train", "val", "val"]})

    dataset = _FakeDataset()
    selected = _select_best_params_full_val(dataset, cfg, spec)
    assert selected["ridge_alpha"] == 0.2

    leaderboard_files = list((tmp_path / "run").glob("full_val_candidate_leaderboard_*.txt"))
    assert len(leaderboard_files) == 1
    body = leaderboard_files[0].read_text(encoding="utf-8")
    assert "0.300000" in body
    assert body.splitlines()[1].startswith("1\t")


def test_select_best_params_cv_path_unchanged(tmp_path: Path, monkeypatch) -> None:
    spec = _wire_pipeline_mocks(monkeypatch, co_bps_sequence=[0.05, 0.05, 0.40, 0.40])

    cfg = ExperimentConfig(
        model_type="test_model",
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
        baseline={"ridge_alpha": 0.1},
        improvement={
            "cv_folds": 2,
            "candidates": [{"ridge_alpha": 0.1}, {"ridge_alpha": 0.9}],
        },
        output_dir=str(tmp_path / "run_cv"),
    )

    class _FakeDataset:
        def __init__(self) -> None:
            self.trial_info = pd.DataFrame({"split": ["train", "train", "val", "val"]})

    dataset = _FakeDataset()
    selected = _select_best_params(dataset, cfg, spec)
    assert selected["ridge_alpha"] == 0.9
    assert not list((tmp_path / "run_cv").glob("full_val_candidate_leaderboard_*.txt"))
