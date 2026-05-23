from __future__ import annotations

import argparse

from nlb_project.public_test import (
    PUBLIC_TEST_EVAL_DATA_SHA256,
    PUBLIC_TEST_EVAL_DATA_SIZE_BYTES,
    download_public_test_eval_data,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public NLB test evaluation targets")
    parser.add_argument(
        "--out",
        default="data/eval/eval_data_test.h5",
        help="Destination HDF5 path",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = download_public_test_eval_data(args.out, force=args.force)
    action = "Downloaded" if result.downloaded else "Already present"
    print(f"{action}: {result.path}")
    print(f"sha256: {result.sha256}")
    print(f"size: {result.size_bytes} bytes")
    if result.sha256 != PUBLIC_TEST_EVAL_DATA_SHA256:
        raise SystemExit("Downloaded file failed sha256 verification")
    if result.size_bytes != PUBLIC_TEST_EVAL_DATA_SIZE_BYTES:
        print(f"WARNING expected {PUBLIC_TEST_EVAL_DATA_SIZE_BYTES} bytes")


if __name__ == "__main__":
    main()
