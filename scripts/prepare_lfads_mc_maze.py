"""Prepare NLB MC_Maze HDF5 for lfads-torch (separate from STNDT-lite training)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from lfads_data_utils import (
    copy_h5,
    default_lfads_torch_dir,
    encod_from_heldin,
    inspect_h5,
    reference_mc_maze_h5,
    repo_root,
    stack_recon,
    subset_h5,
    write_manifest,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-size-ms",
        type=int,
        choices=(5, 20),
        default=20,
        help="Bin width label for output naming (default: 20 = lfads-torch reference)",
    )
    parser.add_argument(
        "--source",
        choices=("reference", "nwb"),
        default="reference",
        help="reference: copy lfads-torch bundled HDF5; nwb: build from NLB NWB via nlb_tools",
    )
    parser.add_argument(
        "--lfads-torch-dir",
        type=Path,
        default=None,
        help="Path to lfads-torch clone (default: LFADS_TORCH_DIR or external/lfads-torch)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="NLB NWB directory (for --source nwb); else NLB_DATA_DIR + mc_maze default",
    )
    parser.add_argument("--data-prefix", default="*full")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HDF5 (default: data/lfads/mc_maze_{bin}ms_val.h5)",
    )
    parser.add_argument(
        "--write-smoke-subset",
        action="store_true",
        help="Also write a tiny trial subset for smoke tests",
    )
    parser.add_argument("--max-train-trials", type=int, default=8)
    parser.add_argument("--max-valid-trials", type=int, default=4)
    return parser.parse_args(argv)


def _default_out(bin_size_ms: int) -> Path:
    return repo_root() / "data" / "lfads" / f"mc_maze_{bin_size_ms}ms_val.h5"


def _prepare_from_reference(lfads_dir: Path, bin_size_ms: int, out_path: Path) -> dict:
    ref = reference_mc_maze_h5(lfads_dir, bin_size_ms)
    if not ref.is_file():
        raise FileNotFoundError(
            f"Reference HDF5 not found: {ref}\n"
            "Run scripts/setup_lfads_torch.ps1 (or .sh) to clone lfads-torch first."
        )
    copy_h5(ref, out_path)
    shapes = inspect_h5(out_path)
    return {
        "source": "lfads-torch reference copy",
        "reference_path": str(ref),
        "bin_size_ms": bin_size_ms,
        "shapes": shapes,
        "task": "held-in encod, held-in+held-out+forward recon (NLB MC_Maze val file)",
    }


def _prepare_from_nwb(
    bin_size_ms: int,
    out_path: Path,
    data_path: str | None,
    data_prefix: str,
) -> dict:
    try:
        from nlb_tools.make_tensors import make_eval_target_tensors, make_train_input_tensors
        from nlb_tools.nwb_interface import NWBDataset
    except ImportError as exc:
        raise SystemExit(
            "nwb_tools is required for --source nwb. "
            "Use the lfads-nlb env or: pip install nlb-tools==0.0.4"
        ) from exc

    sys.path.insert(0, str(repo_root() / "src"))
    from nlb_project.data_contract import resolve_data_path

    dataset_path = resolve_data_path("mc_maze", data_path, data_prefix)
    dataset = NWBDataset(dataset_path, data_prefix, skip_fields=["lick_times", "lick_rates"])
    dataset.resample(bin_size_ms)

    def trial_arrays(trial_split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train = make_train_input_tensors(
            dataset,
            "mc_maze",
            trial_split=trial_split,
            save_file=False,
            include_forward_pred=True,
            include_behavior=True,
        )
        encod = encod_from_heldin(train["train_spikes_heldin"])
        recon = stack_recon(
            train["train_spikes_heldin"],
            train["train_spikes_heldout"],
            train["train_spikes_heldin_forward"],
            train["train_spikes_heldout_forward"],
        )
        behavior = train["train_behavior"]
        return encod, recon, behavior

    targets = make_eval_target_tensors(
        dataset,
        "mc_maze",
        train_trial_split="train",
        eval_trial_split="val",
        save_file=False,
        include_psth=True,
    )
    suf = "" if bin_size_ms == 5 else f"_{bin_size_ms}"
    bundle = targets[f"mc_maze{suf}"]

    train_encod, train_recon, train_behavior = trial_arrays("train")
    valid_encod, valid_recon, valid_behavior = trial_arrays("val")

    import h5py

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5file:
        h5file.create_dataset("train_encod_data", data=train_encod, dtype="float16")
        h5file.create_dataset("train_recon_data", data=train_recon, dtype="float16")
        h5file.create_dataset("valid_encod_data", data=valid_encod, dtype="float16")
        h5file.create_dataset("valid_recon_data", data=valid_recon, dtype="float16")
        h5file.create_dataset("train_behavior", data=train_behavior)
        h5file.create_dataset("valid_behavior", data=valid_behavior)
        for key in (
            "psth",
            "train_cond_idx",
            "eval_cond_idx",
            "train_decode_mask",
            "eval_decode_mask",
        ):
            if key in bundle:
                out_key = key.replace("eval_", "valid_")
                h5file.create_dataset(out_key, data=bundle[key])

    shapes = inspect_h5(out_path)
    return {
        "source": "nlb_tools NWB conversion",
        "dataset_path": dataset_path,
        "bin_size_ms": bin_size_ms,
        "shapes": shapes,
        "task": "held-in encod, held-in+held-out+forward recon",
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    lfads_dir = (args.lfads_torch_dir or default_lfads_torch_dir()).resolve()
    out_path = (args.out or _default_out(args.bin_size_ms)).resolve()

    if args.source == "reference":
        if args.bin_size_ms != 20:
            logger.warning(
                "lfads-torch bundled reference is 20 ms only; forcing bin_size_ms=20 for copy."
            )
        meta = _prepare_from_reference(lfads_dir, 20, out_path)
        meta["bin_size_ms"] = 20
    else:
        meta = _prepare_from_nwb(args.bin_size_ms, out_path, args.data_path, args.data_prefix)

    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    write_manifest(manifest_path, meta)
    logger.info("Wrote %s", out_path)
    logger.info("Manifest %s", manifest_path)
    for key in ("train_encod_data", "train_recon_data", "valid_encod_data", "valid_recon_data"):
        logger.info("  %s %s", key, meta["shapes"].get(key))

    if args.write_smoke_subset:
        smoke_path = out_path.with_name(f"{out_path.stem}_smoke.h5")
        shapes = subset_h5(
            out_path,
            smoke_path,
            max_train_trials=args.max_train_trials,
            max_valid_trials=args.max_valid_trials,
        )
        smoke_manifest = {**meta, "subset_of": str(out_path), "shapes": shapes}
        write_manifest(smoke_path.with_suffix(".manifest.json"), smoke_manifest)
        logger.info("Wrote smoke subset %s", smoke_path)


if __name__ == "__main__":
    # Allow running as script from repo root without package install
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
