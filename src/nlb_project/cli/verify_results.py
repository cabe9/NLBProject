from __future__ import annotations

import argparse

from nlb_project.result_provenance import validate_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify committed benchmark result artifacts against metadata"
    )
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat provenance warnings as failures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_results(args.root)

    for warning in report.warnings:
        print(f"WARNING {warning.path}: {warning.message}")
    for error in report.errors:
        print(f"ERROR {error.path}: {error.message}")

    if report.errors:
        print(
            f"Result provenance check failed: {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings across {report.checked_runs} runs."
        )
        raise SystemExit(1)

    if args.strict_warnings and report.warnings:
        print(
            f"Result provenance check failed: {len(report.warnings)} warnings "
            f"across {report.checked_runs} runs."
        )
        raise SystemExit(1)

    print(
        f"Result provenance check passed: {report.checked_runs} runs checked, "
        f"{len(report.warnings)} warnings."
    )


if __name__ == "__main__":
    main()
