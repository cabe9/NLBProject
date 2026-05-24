from __future__ import annotations

import argparse
import logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a validation-only prediction-averaging ensemble screen"
    )
    parser.add_argument("--config", required=True, help="Path to ensemble-screen YAML config")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # On Windows background launches, PyTorch's DLL loader is more reliable
    # when torch initializes before the NLB/HDF5 stack is imported.
    import torch  # noqa: F401

    from nlb_project.ensemble_screen import load_ensemble_screen_config, run_ensemble_screen

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_ensemble_screen_config(args.config)
    result = run_ensemble_screen(cfg, config_path=args.config)
    print("Ensemble screen finished. Key outputs:")
    print(f"- Metrics: {result['metrics_path']}")
    print(f"- Leaderboard: {result['leaderboard_path']}")
    print(f"- Best ensemble: {result['best_ensemble']['name']}")
    print(f"- Best ensemble mean co-bps: {result['best_ensemble']['co-bps']}")
    print(f"- Passes gate: {result['passes_gate']}")


if __name__ == "__main__":
    main()
