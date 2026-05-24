from __future__ import annotations

import argparse
import logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one mixed ensemble on public NLB test")
    parser.add_argument("--config", required=True, help="Path to ensemble public-test YAML config")
    parser.add_argument(
        "--eval-data",
        default="data/eval/eval_data_test.h5",
        help="Public NLB test target HDF5",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default from config",
    )
    parser.add_argument(
        "--ensemble",
        default=None,
        help="Ensemble name in config, default first ensemble",
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

    # Keep torch first in background Windows launches; see run_ensemble_screen.
    import torch  # noqa: F401

    from nlb_project.ensemble_screen import load_ensemble_screen_config, run_ensemble_public_test

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_ensemble_screen_config(args.config)
    result = run_ensemble_public_test(
        cfg,
        eval_data_path=args.eval_data,
        config_path=args.config,
        output_dir=args.output_dir,
        final_train_trial_split=args.final_train_split,
        ensemble_name=args.ensemble,
    )
    print("Ensemble public test evaluation finished. Key outputs:")
    print(f"- Selected co-bps: {result['selected_metrics'].get('co-bps')}")
    print(f"- Output directory: {result['output_dir']}")
    print(f"- Prediction path: {result['prediction_path']}")


if __name__ == "__main__":
    main()
