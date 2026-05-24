from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from nlb_project.ensemble_screen import (
    _aggregate_repeat_rows,
    _average_prediction_dicts,
    load_ensemble_screen_config,
)


def test_average_prediction_dicts_averages_matching_keys() -> None:
    a = {
        "eval_rates_heldout": np.array([[[1.0, 3.0]]], dtype=np.float32),
        "train_rates_heldout": np.array([[[5.0]]], dtype=np.float32),
    }
    b = {
        "eval_rates_heldout": np.array([[[3.0, 5.0]]], dtype=np.float32),
        "train_rates_heldout": np.array([[[7.0]]], dtype=np.float32),
    }

    out = _average_prediction_dicts([a, b])

    np.testing.assert_allclose(out["eval_rates_heldout"], np.array([[[2.0, 4.0]]]))
    np.testing.assert_allclose(out["train_rates_heldout"], np.array([[[6.0]]]))
    assert out["eval_rates_heldout"].dtype == np.float32


def test_average_prediction_dicts_rejects_mismatched_keys() -> None:
    with pytest.raises(ValueError, match="mismatched keys"):
        _average_prediction_dicts(
            [
                {"eval_rates_heldout": np.ones((1, 1, 1), dtype=np.float32)},
                {"other": np.ones((1, 1, 1), dtype=np.float32)},
            ]
        )


def test_aggregate_repeat_rows_applies_gate() -> None:
    rows = [
        {
            "role": "ensemble_repeat",
            "name": "mixed",
            "seed": 0,
            "co-bps": 0.371,
            "vel R2": 0.9,
            "psth R2": None,
        },
        {
            "role": "ensemble_repeat",
            "name": "mixed",
            "seed": 101,
            "co-bps": 0.372,
            "vel R2": 0.91,
            "psth R2": None,
        },
        {
            "role": "ensemble_repeat",
            "name": "mixed",
            "seed": 202,
            "co-bps": 0.369,
            "vel R2": 0.89,
            "psth R2": None,
        },
    ]

    [aggregate] = _aggregate_repeat_rows(rows, gate_mean=0.3704, min_repeats_above=2)
    params = json.loads(aggregate["params"])

    assert aggregate["role"] == "ensemble_aggregate"
    assert aggregate["co-bps"] == pytest.approx((0.371 + 0.372 + 0.369) / 3)
    assert params["repeats_above_gate"] == 2
    assert params["passes_gate"] is True


def test_load_ensemble_screen_config_validates_unknown_ensemble_recipe(tmp_path: Path) -> None:
    raw = {
        "model_type": "stndt_lite",
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
        "baseline": {"d_model": 192},
        "common_params": {"d_model": 192},
        "recipes": {"anchor": {"n_layers": 4}},
        "seeds": [0],
        "ensembles": {"bad": ["anchor", "missing"]},
        "gate": {"mean_co_bps": 0.3704, "min_repeats_above": 2},
        "output_dir": "results/benchmark_runs/test",
    }
    path = tmp_path / "screen.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown recipes"):
        load_ensemble_screen_config(path)
