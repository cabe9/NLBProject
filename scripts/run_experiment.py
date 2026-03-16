from __future__ import annotations

import argparse
import logging

from nlb_project.config import load_config
from nlb_project.pipeline import run_full_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLB reference + selected experiment")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    result = run_full_experiment(cfg)
    print("Experiment finished. Key outputs:")
    print(f"- Reference co-bps: {result['baseline_metrics'].get('co-bps')}")
    print(f"- Selected co-bps: {result['improved_metrics'].get('co-bps')}")


if __name__ == "__main__":
    main()
