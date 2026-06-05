# NLB MC_Maze — portfolio overview (two tracks)

This repository hosts **two intentional model-family tracks** for NLB'21 `mc_maze`
on the unified [`portfolio-nlb-project`](https://github.com/cabe9/NLBProject/tree/portfolio-nlb-project)
branch. Historical development used separate feature branches; this branch is the
portfolio entry point for both tracks.

## Tracks and scope

| Track | Bin size | Role |
|-------|----------|------|
| **STNDT-lite** | **5 ms** | Primary reproducible neural baseline; validation-gated screens and public-test discipline |
| **LFADS** | **20 ms** | Separate `lfads-torch` baseline; smoke → train/val → export → NLB evaluate |
| **LFADS** | **5 ms** | Same bin width as STNDT-lite; S2 stability config (train/val, two seeds; no public-test) |

**Do not compare co-bps across rows without explicit bin-size labels.**  
5 ms STNDT-lite and 20 ms LFADS use different temporal binning and (for LFADS) the bundled lfads-torch HDF5 contract. 5 ms LFADS and 5 ms STNDT-lite share bin width but differ in model family, data contract, and split (LFADS: train/val only).

## Headline results (local, not live EvalAI)

| Track | Metric context | co-bps | Notes |
|-------|----------------|-------:|-------|
| STNDT-lite mixed ensemble | 5 ms, local **public-test** vs frozen target | **0.3830** | See [`docs/results.md`](results.md) |
| LFADS single-seed | 20 ms, **train/val** (`mc_maze_20_split`) | **0.3606** | 100 epochs, batch 32; not public-test — see [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md) |
| LFADS S2 stability (seeds 0/1) | 5 ms, **train/val** (`mc_maze_split`) | **0.3160** / **0.3154** (mean **~0.3157**) | 30 epochs, batch 8, lr **1e-3**, grad clip **1.0**; stable runs; not public-test — see [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md) |

5 ms LFADS S2 is **below** the 5 ms STNDT-lite public-test headline (**0.3830 co-bps**). Label bin size and split on every comparison.

Frozen historical reference (5 ms leaderboard artifact): **0.3862 co-bps** — not a live target.

## Where to read more

**STNDT-lite track:**

- [`docs/nlb_mc_maze_project_summary.md`](nlb_mc_maze_project_summary.md) — project summary, stopped screens P/Q/R1, diagnostics
- [`docs/results.md`](results.md), [`docs/models.md`](models.md), [`docs/experiment-log.md`](experiment-log.md)
- Configs: `configs/benchmarks/`
- Tracked metrics: `results/benchmark_runs/*/metrics.csv`

**LFADS track:**

- [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md) — env, data prep, export/eval, 20 ms baselines, 5 ms S2 stability screen (seeds 0/1)
- Scripts: `scripts/run_lfads_mc_maze_smoke.py`, `scripts/export_lfads_rates.py`, `scripts/evaluate_lfads_outputs.py`
- Local runs (gitignored): `results/lfads_smoke/<run_id>/` (e.g. `20260603T201838Z` = 20 ms 100-epoch best; `20260605T060424Z` / `20260605T171044Z` = 5 ms S2 seeds 0/1)

## Local-only artifacts

| Path | Track | In git? |
|------|-------|---------|
| `results/benchmark_runs/` (screen metrics) | STNDT | Partially (whitelisted `metrics.csv`) |
| `results/lfads_smoke/` | LFADS | No (gitignored) |
| `data/lfads/`, `external/lfads-torch/` | LFADS | No |
| `AGENTS.md` | Both (agent memory) | No (local ignore) |

## Policy (both tracks)

- **Train/val** for model selection; **public-test** only with explicit approval (STNDT); LFADS has not run public-test.
- No broad hyperparameter sweeps without validation gates.
- STNDT-lite knob screens on the current backbone are **stopped** (see project summary on STNDT branch).
- LFADS epoch scaling past 100 epochs is **not** recommended without a new reason (diminishing returns vs 50→100).

## Branch history

Development previously used separate feature branches (`codex/pc-validated-screen-c-sanitized`
for STNDT-lite, `lfads-baseline-setup` for LFADS). The unified
`portfolio-nlb-project` branch merges both; use it as the single portfolio entry point.
