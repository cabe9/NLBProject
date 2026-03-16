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
    model_type = df["model_type"].iloc[0] if "model_type" in df.columns and not df.empty else "unknown"

    def _fmt(value: object) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}"

    lines = [
        "# NLB MC_Maze Summary",
        "",
        f"Model family: `{model_type}`",
        "",
        "| run | co-bps | vel R2 | psth R2 | params |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row.get('co-bps'))} | {_fmt(row.get('vel R2'))} | {_fmt(row.get('psth R2'))} | {row.get('params', '')} |"
        )
    selected = df[df["model"] == "improved"].iloc[0]
    reference = df[df["model"] == "baseline"].iloc[0]
    delta = float(selected["co-bps"]) - float(reference["co-bps"])
    lines.extend(["", f"Delta co-bps (selected - reference): **{delta:.4f}**"])
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
