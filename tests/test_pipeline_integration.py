"""Pipeline-level integration test with synthetic tensors.

This verifies config -> run -> artifact wiring without downloading NLB data.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from nlb_project.config import ExperimentConfig
from nlb_project.model_registry import ModelSpec, SweepAxis

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

from nlb_project.pipeline import run_full_experiment


def test_run_full_experiment_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    """A synthetic end-to-end run writes predictions + metrics artifacts."""

    class _FakeDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            self.trial_info = pd.DataFrame(
                {"split": ["train", "train", "val", "val", "train", "val"]}
            )

        def resample(self, _bin_size_ms: int) -> None:
            return None

    def _fake_train_tensors(_dataset, _dataset_name, trial_split, save_file=False):
        n = int(np.sum(trial_split)) if isinstance(trial_split, np.ndarray) else 4
        return {
            "train_spikes_heldin": np.ones((n, 5, 3), dtype=float),
            "train_spikes_heldout": np.ones((n, 5, 2), dtype=float),
        }

    def _fake_eval_tensors(_dataset, _dataset_name, trial_split, save_file=False):
        n = int(np.sum(trial_split)) if isinstance(trial_split, np.ndarray) else 3
        return {"eval_spikes_heldin": np.ones((n, 5, 3), dtype=float)}

    def _fake_target_tensors(
        _dataset,
        _dataset_name,
        train_trial_split,
        eval_trial_split,
        save_file=False,
        include_psth=False,
    ):
        n_train = int(np.sum(train_trial_split)) if isinstance(train_trial_split, np.ndarray) else 4
        n_eval = int(np.sum(eval_trial_split)) if isinstance(eval_trial_split, np.ndarray) else 3
        return {
            "mc_maze": {
                "eval_spikes_heldout": np.ones((n_eval, 5, 2), dtype=float),
                "train_behavior": np.zeros((n_train, 5, 2), dtype=float),
                "eval_behavior": np.zeros((n_eval, 5, 2), dtype=float),
            }
        }

    def _fake_predict(train_hi, train_ho, eval_hi, *, ridge_alpha, output_head, log_offset):
        n_eval, t, n_hi = eval_hi.shape
        n_ho = train_ho.shape[2]
        return {
            "train_rates_heldin": np.full(train_hi.shape, 0.2, dtype=float),
            "train_rates_heldout": np.full(train_ho.shape, 0.2, dtype=float),
            "eval_rates_heldin": np.full((n_eval, t, n_hi), 0.2, dtype=float),
            "eval_rates_heldout": np.full((n_eval, t, n_ho), 0.2, dtype=float),
        }

    def _fake_evaluate(_target_dict, _output_dict):
        return [{"mc_maze_split": {"co-bps": 0.02, "vel R2": 0.1, "psth R2": None}}]

    def _fake_save_to_h5(output_dict, out_path: str, overwrite=False):
        Path(out_path).write_bytes(repr(sorted(output_dict.keys())).encode("utf-8"))

    spec = ModelSpec(
        name="test_model",
        predict=_fake_predict,
        baseline_params=(("ridge_alpha", float),),
        sweep_axes=(SweepAxis("ridge_alpha_grid", "ridge_alpha", float),),
        uses_rate_head=True,
    )

    monkeypatch.setattr(
        "nlb_project.pipeline.resolve_data_path", lambda *_args, **_kwargs: str(tmp_path)
    )
    monkeypatch.setattr("nlb_project.pipeline.NWBDataset", _FakeDataset)
    monkeypatch.setattr("nlb_project.pipeline.make_train_input_tensors", _fake_train_tensors)
    monkeypatch.setattr("nlb_project.pipeline.make_eval_input_tensors", _fake_eval_tensors)
    monkeypatch.setattr("nlb_project.pipeline.make_eval_target_tensors", _fake_target_tensors)
    monkeypatch.setattr("nlb_project.pipeline.evaluate", _fake_evaluate)
    monkeypatch.setattr("nlb_project.pipeline.save_to_h5", _fake_save_to_h5)
    monkeypatch.setattr("nlb_project.pipeline.get_spec", lambda _model_type: spec)

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
        improvement={"ridge_alpha_grid": [0.1, 1.0], "cv_folds": 2},
        output_dir=str(tmp_path / "run"),
        output_head="log_link",
    )

    result = run_full_experiment(cfg)

    out_dir = tmp_path / "run"
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "ablation.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "predictions" / "baseline_predictions.h5").exists()
    assert (out_dir / "predictions" / "improved_predictions.h5").exists()
    assert result["baseline_metrics"]["co-bps"] == 0.02
    assert result["improved_metrics"]["co-bps"] == 0.02

    with (out_dir / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["model"] for row in rows] == ["baseline", "improved"]
