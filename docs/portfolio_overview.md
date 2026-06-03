# NLB MC_Maze — portfolio overview (two tracks)

This repository hosts **two intentional model-family tracks** for NLB'21 `mc_maze`.
They live on **separate git branches** today; do not merge them without a deliberate
integration plan.

## Branches and scope

| Track | Branch | Bin size | Role |
|-------|--------|----------|------|
| **STNDT-lite** | [`codex/pc-validated-screen-c-sanitized`](https://github.com/cabe9/NLBProject/tree/codex/pc-validated-screen-c-sanitized) | **5 ms** | Primary reproducible neural baseline; validation-gated screens and public-test discipline |
| **LFADS** | [`lfads-baseline-setup`](https://github.com/cabe9/NLBProject/tree/lfads-baseline-setup) | **20 ms** | Separate `lfads-torch` baseline; smoke → train/val → export → NLB evaluate |

**Do not compare co-bps across rows without explicit bin-size labels.**  
5 ms STNDT-lite and 20 ms LFADS use different temporal binning and (for LFADS) the bundled lfads-torch HDF5 contract.

## Headline results (local, not live EvalAI)

| Track | Metric context | co-bps | Notes |
|-------|----------------|-------:|-------|
| STNDT-lite mixed ensemble | 5 ms, local **public-test** vs frozen target | **0.3830** | Headline on STNDT branch; see [`docs/results.md`](results.md) (STNDT track; also in this tree) |
| LFADS single-seed | 20 ms, **train/val** (`mc_maze_20_split`) | **0.3606** | 100 epochs, batch 32; not public-test — see [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md) |

Frozen historical reference (5 ms leaderboard artifact): **0.3862 co-bps** — not a live target.

## Where to read more

**STNDT branch** (`codex/pc-validated-screen-c-sanitized`) — check out that branch for the canonical STNDT workflow:

- `docs/nlb_mc_maze_project_summary.md` — project summary, stopped screens P/Q/R1, diagnostics (**available only on branch `codex/pc-validated-screen-c-sanitized`**, not on `lfads-baseline-setup`)
- [`docs/results.md`](results.md), [`docs/models.md`](models.md), [`docs/experiment-log.md`](experiment-log.md) — also present in this checkout; STNDT configs and screen history are authoritative on the STNDT branch
- Configs: `configs/benchmarks/` (on STNDT branch; may differ from this branch over time)
- Tracked metrics: `results/benchmark_runs/*/metrics.csv`

**LFADS branch** (`lfads-baseline-setup`):

- [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md) — env, data prep, export/eval, completed 2/50/100-epoch runs
- Scripts: `scripts/run_lfads_mc_maze_smoke.py`, `scripts/export_lfads_rates.py`, `scripts/evaluate_lfads_outputs.py`
- Local runs (gitignored): `results/lfads_smoke/<run_id>/` (e.g. `20260603T201838Z` = 100-epoch best)

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

## Future integration (optional)

A later **`portfolio-nlb-project`** (or similar) branch could merge:

- STNDT headline + screen history + diagnostics docs
- LFADS baseline docs and scripts
- This overview as the single entry point

Until then, treat each branch as **branch-specific truth** for its track and use this file (once present on both branches) as the cross-link.
