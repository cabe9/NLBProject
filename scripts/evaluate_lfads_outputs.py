"""Evaluate LFADS rate exports with the NLB-style validator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lfads-output-h5",
        type=Path,
        default=None,
        help="LFADS posterior HDF5 (train/valid_output_params)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Smoke run dir; uses lfads_outputs/*.h5 if --lfads-output-h5 omitted",
    )
    parser.add_argument(
        "--data-h5",
        type=Path,
        default=None,
        help="LFADS data HDF5 for targets (defaults to run manifest data_h5)",
    )
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--dataset-name", default="mc_maze")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report (default: next to lfads output h5)",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only alignment checks, do not call nlb_tools.evaluate",
    )
    return parser.parse_args(argv)


def _resolve_lfads_output(run_dir: Path | None, lfads_output: Path | None) -> Path:
    if lfads_output is not None:
        path = lfads_output.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if run_dir is None:
        raise ValueError("Provide --lfads-output-h5 or --run-dir")
    run_dir = run_dir.expanduser().resolve()
    out_dir = run_dir / "lfads_outputs"
    candidates = sorted(out_dir.glob("lfads_output_*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No lfads_output_*.h5 under {out_dir}")
    return candidates[-1]


def _resolve_data_h5(
    data_h5: Path | None,
    run_dir: Path | None,
    lfads_output: Path,
) -> Path:
    if data_h5 is not None:
        return data_h5.expanduser().resolve()
    if run_dir is not None:
        manifest_path = run_dir.expanduser().resolve() / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("data_h5"):
                return Path(manifest["data_h5"]).resolve()
    # LFADS output is a copy of data file + output_params
    return lfads_output


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, str(_repo_root() / "scripts"))

    from lfads_nlb_bridge import (
        dims_from_lfads_h5,
        nlb_dataset_key,
        nlb_split_key,
        run_nlb_evaluate,
        targets_from_lfads_data_h5,
        user_dict_from_lfads_output_h5,
        verify_nlb_alignment,
    )

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    lfads_path = _resolve_lfads_output(run_dir, args.lfads_output_h5)
    data_h5 = _resolve_data_h5(args.data_h5, run_dir, lfads_path)

    dims = dims_from_lfads_h5(data_h5)
    user = user_dict_from_lfads_output_h5(
        lfads_path,
        dataset_name=args.dataset_name,
        bin_size_ms=args.bin_size_ms,
        data_h5=data_h5,
    )
    target = targets_from_lfads_data_h5(
        data_h5,
        dataset_name=args.dataset_name,
        bin_size_ms=args.bin_size_ms,
    )
    alignment = verify_nlb_alignment(
        user,
        target,
        dataset_name=args.dataset_name,
        bin_size_ms=args.bin_size_ms,
    )

    dkey = nlb_dataset_key(args.dataset_name, args.bin_size_ms)
    psth_included = "psth" in target[dkey]
    report: dict = {
        "status": "alignment_only",
        "bin_size_ms": args.bin_size_ms,
        "bin_size_label": (
            f"{args.bin_size_ms} ms LFADS — not comparable to 5 ms STNDT-lite headline (0.3830)"
        ),
        "lfads_output_h5": str(lfads_path),
        "data_h5": str(data_h5),
        "dataset_key": dkey,
        "split_key": nlb_split_key(args.dataset_name, args.bin_size_ms),
        "dims": dims,
        "lfads_output_shapes": {
            k: list(v.shape)
            for k, v in {
                "train_output_params": _load_if_present(lfads_path, "train_output_params"),
                "valid_output_params": _load_if_present(lfads_path, "valid_output_params"),
            }.items()
            if v is not None
        },
        "nlb_user_shapes": {k: list(v.shape) for k, v in user[dkey].items()},
        "nlb_target_shapes": {k: list(v.shape) for k, v in target[dkey].items()},
        "alignment": alignment,
        "evaluator_ready": alignment["aligned"],
        "psth_included": psth_included,
        "psth_note": (
            None
            if psth_included
            else "Omitted: trial-subset HDF5 has cond_idx pointing outside subset trials"
        ),
        "split_mapping": {
            "lfads_valid_output_params": "NLB eval_rates_* (validation / val trials)",
            "lfads_train_output_params": "NLB train_rates_* (train trials)",
            "valid_recon_data[:, :tlen, n_heldin:]": "eval_spikes_heldout target for co-bps",
        },
    }

    metrics = None
    eval_meta = {}
    if not args.skip_metrics and alignment["aligned"]:
        results, eval_meta = run_nlb_evaluate(user, target)
        report["evaluate_meta"] = eval_meta
        if results is not None:
            split_key = report["split_key"]
            for block in results:
                if split_key in block:
                    metrics = {k: float(v) if np.isfinite(v) else v for k, v in block[split_key].items()}
                    break
            report["metrics"] = metrics
            report["status"] = "evaluated" if metrics else "evaluate_no_split"
        else:
            report["status"] = "evaluate_blocked"
            report["missing"] = eval_meta.get("reason", "unknown")
    elif not alignment["aligned"]:
        report["status"] = "misaligned"
        report["missing"] = "Fix shape/trial alignment before calling evaluate()"
    else:
        report["status"] = "alignment_only"

    if metrics is None and report["status"] not in ("evaluated",):
        report.setdefault(
            "missing_for_full_nlb_score",
            [
                "Posterior export must produce finite train/valid_output_params.",
                "Targets from data HDF5 must match val trial counts on rate tensors.",
                "psth R2 may be skipped on smoke subsets when psth (108 cond) "
                "does not match trial cond_idx length.",
            ],
        )

    out_path = args.out or lfads_path.with_name(lfads_path.stem + "_nlb_eval.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    if metrics:
        co = metrics.get("co-bps")
        if co is not None:
            print(f"\nValidation co-bps ({args.bin_size_ms} ms): {co:.4f}")
            print("(Diagnostic smoke score — not comparable to 5 ms STNDT-lite 0.3830.)")
    elif report.get("status") == "evaluated":
        print("\nNo metrics block returned.")
    else:
        print("\nNo co-bps reported.", report.get("missing") or report.get("missing_for_full_nlb_score"))
    return 0 if report.get("evaluator_ready") else 1


def _load_if_present(h5_path: Path, key: str) -> np.ndarray | None:
    import h5py

    with h5py.File(h5_path, "r") as h5file:
        if key not in h5file:
            return None
        return np.asarray(h5file[key][()], dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
