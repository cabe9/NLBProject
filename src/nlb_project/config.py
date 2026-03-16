from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    dataset_name: str
    data_path: str | None
    data_prefix: str
    bin_size_ms: int
    train_split: str
    eval_split: str
    include_psth: bool
    log_offset: float
    seed: int
    skip_fields: list[str]
    baseline: dict[str, Any]
    improvement: dict[str, Any]
    output_dir: str
    model_type: str = "smoothing"


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)
