from __future__ import annotations

import os
from glob import glob
from pathlib import Path


_DEFAULT_DATASET_SUBPATH = {
    "mc_maze": "000128/sub-Jenkins",
    "mc_rtt": "000129/sub-Indy",
    "area2_bump": "000127/sub-Han",
    "dmfc_rsg": "000130/sub-Haydn",
    "mc_maze_large": "000138/sub-Jenkins",
    "mc_maze_medium": "000139/sub-Jenkins",
    "mc_maze_small": "000140/sub-Jenkins",
}


def resolve_data_path(dataset_name: str, data_path: str | None, data_prefix: str) -> str:
    if data_path:
        candidate = Path(data_path).expanduser().resolve()
    else:
        root = os.environ.get("NLB_DATA_DIR")
        if not root:
            raise ValueError(
                "Data path not provided. Set `data_path` in config or set `NLB_DATA_DIR` "
                "to a root directory containing NLB datasets (e.g. data/raw)."
            )
        if dataset_name not in _DEFAULT_DATASET_SUBPATH:
            raise ValueError(f"No default path mapping for dataset: {dataset_name}")
        candidate = Path(root).expanduser().resolve() / _DEFAULT_DATASET_SUBPATH[dataset_name]

    if not candidate.exists():
        raise FileNotFoundError(f"NLB data path not found: {candidate}")

    pattern = str(candidate / f"{data_prefix}*.nwb")
    if len(glob(pattern)) == 0:
        raise FileNotFoundError(
            f"No NWB files matched pattern `{pattern}`. "
            "Verify your dataset path and data_prefix."
        )
    return str(candidate)
