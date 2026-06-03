"""Tests for LFADS -> NLB conversion (no lfads-torch import)."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lfads_nlb_bridge import (  # noqa: E402
    dims_from_lfads_h5,
    nlb_dataset_key,
    split_recon_rates,
    targets_from_lfads_data_h5,
    user_dict_from_lfads_output_h5,
    verify_nlb_alignment,
)


def _write_lfads_pair(path: Path, n_train: int = 6, n_valid: int = 3) -> None:
    rng = np.random.default_rng(0)
    n_hi, n_ho, tlen, fp = 3, 2, 5, 2
    recon_t = tlen + fp
    n_ch = n_hi + n_ho

    def recon(n):
        main = rng.random((n, tlen, n_ch), dtype=np.float32)
        fwd = rng.random((n, fp, n_ch), dtype=np.float32)
        return np.concatenate([main, fwd], axis=1).astype(np.float16)

    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("train_encod_data", data=rng.random((n_train, tlen, n_hi)))
        h5file.create_dataset("valid_encod_data", data=rng.random((n_valid, tlen, n_hi)))
        h5file.create_dataset("train_recon_data", data=recon(n_train))
        h5file.create_dataset("valid_recon_data", data=recon(n_valid))
        h5file.create_dataset("train_behavior", data=rng.random((n_train, tlen, 2)))
        h5file.create_dataset("valid_behavior", data=rng.random((n_valid, tlen, 2)))
        h5file.create_dataset("train_output_params", data=recon(n_train).astype(np.float32))
        h5file.create_dataset("valid_output_params", data=recon(n_valid).astype(np.float32))


def test_split_recon_rates_shapes() -> None:
    rates = np.ones((4, 7, 5))
    parts = split_recon_rates(rates, n_heldin=3, tlen=5)
    assert parts["rates_heldin"].shape == (4, 5, 3)
    assert parts["rates_heldout"].shape == (4, 5, 2)
    assert parts["rates_heldin_forward"].shape == (4, 2, 3)


def test_bridge_roundtrip_and_alignment(tmp_path: Path) -> None:
    h5 = tmp_path / "lfads.h5"
    _write_lfads_pair(h5)
    dims = dims_from_lfads_h5(h5)
    assert dims["n_heldin"] == 3
    assert dims["tlen"] == 5

    user = user_dict_from_lfads_output_h5(h5, bin_size_ms=20, data_h5=h5)
    target = targets_from_lfads_data_h5(h5, bin_size_ms=20)
    key = nlb_dataset_key("mc_maze", 20)
    assert user[key]["eval_rates_heldout"].shape == target[key]["eval_spikes_heldout"].shape

    report = verify_nlb_alignment(user, target, bin_size_ms=20)
    assert report["aligned"] is True
