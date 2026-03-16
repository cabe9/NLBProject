from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_metrics_csv(rows: list[dict], out_path: str | Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def write_summary_md(rows: list[dict], out_path: str | Path) -> None:
    df = pd.DataFrame(rows)
    lines = ["# NLB MC_Maze Summary", "", "| model | co-bps | vel R2 | psth R2 |", "|---|---:|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row.get('co-bps', float('nan')):.4f} | {row.get('vel R2', float('nan')):.4f} | {row.get('psth R2', float('nan')):.4f} |"
        )
    improved = df[df["model"] == "improved"].iloc[0]
    baseline = df[df["model"] == "baseline"].iloc[0]
    delta = improved["co-bps"] - baseline["co-bps"]
    lines.extend(["", f"Delta co-bps (improved - baseline): **{delta:.4f}**"])
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
