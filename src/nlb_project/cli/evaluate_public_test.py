from __future__ import annotations

import argparse
import logging

from nlb_project.config import load_config
from nlb_project.public_test import run_public_test_evaluation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a config on public NLB test targets")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--eval-data",
        default="data/eval/eval_data_test.h5",
        help="Public NLB test target HDF5",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default results/public_test/<dataset_name>",
    )
    parser.add_argument(
        "--final-train-split",
        nargs="+",
        default=["train", "val"],
        help="Trial split(s) used for the final public-test fit",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    result = run_public_test_evaluation(
        cfg,
        eval_data_path=args.eval_data,
        config_path=args.config,
        output_dir=args.output_dir,
        final_train_trial_split=args.final_train_split,
    )
    print("Public test evaluation finished. Key outputs:")
    print(f"- Baseline co-bps: {result['baseline_metrics'].get('co-bps')}")
    print(f"- Selected co-bps: {result['selected_metrics'].get('co-bps')}")
    print(f"- Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
