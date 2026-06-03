"""Convert LFADS posterior outputs to NLB-style prediction/target dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

MC_MAZE_HELDIN_20MS = 137
MC_MAZE_HELDOUT_20MS = 45
MC_MAZE_TLEN_20MS = 35


def nlb_dataset_key(dataset_name: str, bin_size_ms: int) -> str:
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    return f"{dataset_name}{suf}"


def nlb_split_key(dataset_name: str, bin_size_ms: int) -> str:
    return f"{nlb_dataset_key(dataset_name, bin_size_ms)}_split"


def dims_from_lfads_h5(data_h5: Path) -> dict[str, int]:
    """Infer held-in / time dimensions from an LFADS MC_Maze HDF5."""
    with h5py.File(data_h5, "r") as h5file:
        n_heldin = int(h5file["train_encod_data"].shape[-1])
        tlen = int(h5file["train_encod_data"].shape[1])
        recon_seq_len = int(h5file["train_recon_data"].shape[1])
        n_channels = int(h5file["train_recon_data"].shape[-1])
    return {
        "n_heldin": n_heldin,
        "n_heldout": n_channels - n_heldin,
        "tlen": tlen,
        "recon_seq_len": recon_seq_len,
        "fp_steps": recon_seq_len - tlen,
    }


def split_recon_rates(
    rates: np.ndarray,
    *,
    n_heldin: int,
    tlen: int,
) -> dict[str, np.ndarray]:
    """Split combined recon rates (trials x recon_time x channels) into NLB fields."""
    rates = np.asarray(rates, dtype=np.float64)
    heldin = rates[:, :tlen, :n_heldin]
    heldout = rates[:, :tlen, n_heldin:]
    forward = rates[:, tlen:, :]
    return {
        "rates_heldin": heldin,
        "rates_heldout": heldout,
        "rates_heldin_forward": forward[:, :, :n_heldin],
        "rates_heldout_forward": forward[:, :, n_heldin:],
    }


def spikes_from_recon(recon: np.ndarray, *, n_heldin: int, tlen: int) -> dict[str, np.ndarray]:
    """Extract spike-count targets from LFADS recon tensors (observed + forward)."""
    recon = np.asarray(recon, dtype=np.float64)
    return {
        "spikes_heldin": recon[:, :tlen, :n_heldin],
        "spikes_heldout": recon[:, :tlen, n_heldin:],
        "spikes_heldin_forward": recon[:, tlen:, :n_heldin],
        "spikes_heldout_forward": recon[:, tlen:, n_heldin:],
    }


def targets_from_lfads_data_h5(
    data_h5: Path,
    *,
    dataset_name: str = "mc_maze",
    bin_size_ms: int = 20,
) -> dict[str, dict[str, np.ndarray]]:
    """Build NLB evaluator target dict from LFADS train/valid recon tensors."""
    dims = dims_from_lfads_h5(data_h5)
    n_hi = dims["n_heldin"]
    tlen = dims["tlen"]
    key = nlb_dataset_key(dataset_name, bin_size_ms)

    with h5py.File(data_h5, "r") as h5file:
        train_recon = np.asarray(h5file["train_recon_data"][()], dtype=np.float64)
        valid_recon = np.asarray(h5file["valid_recon_data"][()], dtype=np.float64)
        train_sp = spikes_from_recon(train_recon, n_heldin=n_hi, tlen=tlen)
        valid_sp = spikes_from_recon(valid_recon, n_heldin=n_hi, tlen=tlen)

        bundle: dict[str, np.ndarray] = {
            "train_spikes_heldin": train_sp["spikes_heldin"],
            "train_spikes_heldout": train_sp["spikes_heldout"],
            "eval_spikes_heldin": valid_sp["spikes_heldin"],
            "eval_spikes_heldout": valid_sp["spikes_heldout"],
            "train_behavior": np.asarray(h5file["train_behavior"][()], dtype=np.float64),
            "eval_behavior": np.asarray(h5file["valid_behavior"][()], dtype=np.float64),
            "eval_spikes_heldin_forward": valid_sp["spikes_heldin_forward"],
            "eval_spikes_heldout_forward": valid_sp["spikes_heldout_forward"],
        }
        for mask_key, lfads_key in (
            ("train_decode_mask", "train_decode_mask"),
            ("eval_decode_mask", "valid_decode_mask"),
        ):
            if lfads_key in h5file:
                bundle[mask_key] = np.asarray(h5file[lfads_key][()])

        if "psth" in h5file and "valid_cond_idx" in h5file:
            cond_idx = np.asarray(h5file["valid_cond_idx"][()])
            if _psth_indices_in_range(cond_idx, n_trials=valid_recon.shape[0]):
                bundle["psth"] = np.asarray(h5file["psth"][()], dtype=np.float64)
                bundle["eval_cond_idx"] = cond_idx

    return {key: bundle}


def _psth_indices_in_range(cond_idx: np.ndarray, *, n_trials: int) -> bool:
    """True when condition index lists only reference trials present in a subset."""
    try:
        for entry in cond_idx:
            idx = np.asarray(entry).ravel()
            if idx.size and int(idx.max()) >= n_trials:
                return False
        return True
    except (TypeError, ValueError):
        return False


def user_dict_from_lfads_output_h5(
    lfads_output_h5: Path,
    *,
    dataset_name: str = "mc_maze",
    bin_size_ms: int = 20,
    data_h5: Path | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Map LFADS `train/valid_output_params` to NLB submission rate keys."""
    ref = data_h5 or lfads_output_h5
    dims = dims_from_lfads_h5(ref)
    key = nlb_dataset_key(dataset_name, bin_size_ms)

    with h5py.File(lfads_output_h5, "r") as h5file:
        if "train_output_params" not in h5file or "valid_output_params" not in h5file:
            raise KeyError(
                f"{lfads_output_h5} missing train_output_params or valid_output_params; "
                "run scripts/export_lfads_rates.py first."
            )
        train_rates = np.asarray(h5file["train_output_params"][()], dtype=np.float64)
        valid_rates = np.asarray(h5file["valid_output_params"][()], dtype=np.float64)

    train_split = split_recon_rates(
        train_rates, n_heldin=dims["n_heldin"], tlen=dims["tlen"]
    )
    valid_split = split_recon_rates(
        valid_rates, n_heldin=dims["n_heldin"], tlen=dims["tlen"]
    )

    return {
        key: {
            "train_rates_heldin": train_split["rates_heldin"],
            "train_rates_heldout": train_split["rates_heldout"],
            "eval_rates_heldin": valid_split["rates_heldin"],
            "eval_rates_heldout": valid_split["rates_heldout"],
            "eval_rates_heldin_forward": valid_split["rates_heldin_forward"],
            "eval_rates_heldout_forward": valid_split["rates_heldout_forward"],
        }
    }


def verify_nlb_alignment(
    user: dict[str, dict[str, np.ndarray]],
    target: dict[str, dict[str, np.ndarray]],
    *,
    dataset_name: str = "mc_maze",
    bin_size_ms: int = 20,
) -> dict[str, Any]:
    """Check shapes and trial alignment between user rates and targets."""
    key = nlb_dataset_key(dataset_name, bin_size_ms)
    u = user[key]
    t = target[key]
    report: dict[str, Any] = {
        "dataset_key": key,
        "split_key": nlb_split_key(dataset_name, bin_size_ms),
        "aligned": True,
        "checks": {},
    }

    rate_spike_pairs = [
        ("eval_rates_heldout", "eval_spikes_heldout"),
        ("eval_rates_heldin", "eval_spikes_heldin"),
        ("train_rates_heldout", "train_spikes_heldout"),
        ("train_rates_heldin", "train_spikes_heldin"),
    ]
    for user_k, target_k in rate_spike_pairs:
        if user_k not in u or target_k not in t:
            continue
        us = u[user_k].shape
        ts = t[target_k].shape
        ok = us == ts
        report["checks"][f"{user_k}_vs_{target_k}"] = {
            "user_shape": list(us),
            "target_shape": list(ts),
            "ok": bool(ok),
        }
        if not ok:
            report["aligned"] = False

    for user_k, target_k in (("eval_rates_heldin", "eval_behavior"), ("train_rates_heldin", "train_behavior")):
        if user_k not in u or target_k not in t:
            continue
        us = u[user_k].shape
        ts = t[target_k].shape
        ok = us[0] == ts[0] and us[1] == ts[1]
        report["checks"][f"{user_k}_vs_{target_k}"] = {
            "user_shape": list(us),
            "target_shape": list(ts),
            "ok": bool(ok),
        }
        if not ok:
            report["aligned"] = False

    ho = u.get("eval_rates_heldout")
    hi = u.get("eval_rates_heldin")
    if ho is not None and hi is not None:
        report["eval_rate_ranges"] = {
            "heldin_min": float(hi.min()),
            "heldin_max": float(hi.max()),
            "heldout_min": float(ho.min()),
            "heldout_max": float(ho.max()),
            "heldout_finite": bool(np.isfinite(ho).all()),
        }
    return report


def run_nlb_evaluate(
    user: dict[str, dict[str, np.ndarray]],
    target: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Run nlb_tools.evaluate when requirements are met."""
    meta: dict[str, Any] = {"evaluator": "nlb_tools.evaluation.evaluate"}
    key = next(iter(user.keys()))
    if "eval_rates_heldout" not in user[key]:
        meta["status"] = "blocked"
        meta["reason"] = "user dict missing eval_rates_heldout"
        return None, meta
    if "eval_spikes_heldout" not in target.get(key, {}):
        meta["status"] = "blocked"
        meta["reason"] = "target dict missing eval_spikes_heldout"
        return None, meta

    from nlb_tools.evaluation import evaluate

    try:
        results = evaluate(target, user)
    except IndexError as exc:
        meta["status"] = "evaluate_error"
        meta["reason"] = (
            f"nlb_tools.evaluate failed ({exc}). "
            "Likely psth/cond_idx mismatch on a trial subset — "
            "targets omit psth when cond indices exceed trial count."
        )
        return None, meta
    split_key = f"{key}_split"
    metrics = None
    for block in results:
        if split_key in block:
            metrics = block[split_key]
            break
    meta["status"] = "ok" if metrics is not None else "partial"
    if metrics is None:
        meta["reason"] = f"split key {split_key} not in evaluate() output"
        meta["result_keys"] = [list(b.keys()) for b in results]
    return results, meta
