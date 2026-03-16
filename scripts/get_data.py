from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_DATASET_URLS = {
    "mc_maze": "https://dandiarchive.org/dandiset/000128/draft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NLB dataset(s) from DANDI")
    parser.add_argument("--dataset", default="mc_maze", choices=sorted(_DATASET_URLS.keys()))
    parser.add_argument("--out", default="data/raw", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("dandi") is None:
        print("`dandi` CLI not found. Install with: pip install dandi", file=sys.stderr)
        sys.exit(1)

    url = _DATASET_URLS[args.dataset]
    cmd = ["dandi", "download", "-o", str(out_dir), url]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\nData download complete.")
    print(f"Set environment variable for runner: export NLB_DATA_DIR={out_dir}")


if __name__ == "__main__":
    main()
