# Results

This repo evaluates NLB'21 `mc_maze` locally against the public test-target
HDF5 from the official `nlb_tools` codepack. EvalAI submissions are closed, so
these scores are reproducible local public-test scores, not live leaderboard
ranks.

## Headline

The current best model is a validation-selected mixed `stndt_lite` ensemble:
a 4-layer `learning_rate=0.0013` 5-seed ensemble averaged with a 5-layer
`dropout=0.08` 5-seed ensemble. Both components keep the mask 0.6 objective,
full held-in loss weight, mask token, temporal neuron-event identity, and
spike-weighted masked reconstruction.

| Model | co-bps | vel R² | psth R² |
|---|---:|---:|---:|
| lagged PCA latent regression, selected history | 0.0268 | 0.3678 | -26.4081 |
| NDT-lite, stability-tuned 192-wide 7-seed ensemble | 0.3229 | 0.7693 | 0.6368 |
| STNDT-lite, spatiotemporal 5-seed ensemble | 0.3302 | 0.8138 | 0.6441 |
| STNDT-lite, temporal identity + spike-weighted 5-seed ensemble | 0.3413 | 0.8451 | -1.8622 |
| STNDT-lite, Screen C CD-reconcile winner 5-seed ensemble | 0.3649 | 0.8911 | 0.6548 |
| STNDT-lite, 4-layer constant mask 0.5 5-seed ensemble | 0.3742 | 0.8949 | 0.6566 |
| STNDT-lite, 4-layer constant mask 0.6 5-seed ensemble | 0.3795 | 0.8978 | 0.6354 |
| **STNDT-lite, mixed lr0.0013/depth5 10-member ensemble** | **0.3830** | **0.9053** | **0.6390** |

## Context

The lagged PCA baseline is a reproducible floor. The current neural result is
about `+0.356 co-bps` above that baseline and above the prior 4-layer mask 0.6
public-test level (`0.3795 co-bps`).

The frozen `MC_Maze 5 ms` rank-1 leaderboard reference is `0.3862 co-bps`
(`STNDT[Ensemble]`). This repo's current `0.3830 co-bps` result is a meaningful
local benchmark improvement, but it should not be described as a live
leaderboard rank.

## LFADS baselines (separate track, `lfads-baseline-setup` branch)

LFADS scores are **train/val only** via `nlb_tools.evaluate` — **not public-test**.
Label bin size on every score.

| Model | Bin size | Split | co-bps | Notes |
|---|---|---|---:|---|
| LFADS (`lfads-torch`), 100-epoch single seed | 20 ms | train/val | **0.3606** | Bundled lfads-torch HDF5 |
| LFADS S2 stability, seed 0 | 5 ms | train/val | **0.3160** | 30 epochs, lr 1e-3, grad clip 1.0, batch 8 |
| LFADS S2 stability, seed 1 | 5 ms | train/val | **0.3154** | Same config as seed 0 |
| LFADS S2 stability, two-seed mean | 5 ms | train/val | **~0.3157** | Reproducible stable 30-epoch runs |

5 ms LFADS S2 (**~0.3157 co-bps** train/val) is **below** the STNDT-lite 5 ms
public-test headline (**0.3830 co-bps**). Do not compare without bin-size and
split labels. Details: [`docs/lfads_baseline_plan.md`](lfads_baseline_plan.md).

## Extended model metrics

Saved scores from local artifacts only. Every row is labeled by **bin size** and
**split/type**. LFADS rows are **train/val** (`nlb_tools.evaluate` on LFADS HDF5
targets); STNDT-lite public-test rows are **local public-test** (frozen NLB test
target HDF5). Train/val LFADS scores are **not** equivalent to STNDT public-test
scores.

| model family | run/config | bin size | split/type | co-bps | fp-bps | vel R² | psth R² | notes |
|---|---|---|---|---:|---:|---:|---:|---|
| STNDT-lite | mixed lr0013+depth5 10-member ensemble (headline) | 5 ms | public-test | **0.3830** | not computed | 0.9053 | 0.6390 | `mc_maze_stndt_lite_diverse_lr0013_depth5_10member` |
| STNDT-lite | 4-layer mask 0.6 5-seed | 5 ms | public-test | 0.3795 | not computed | 0.8978 | 0.6354 | headline ensemble component |
| STNDT-lite | 5-layer dropout 0.08 5-seed | 5 ms | public-test | 0.3764 | not computed | 0.9026 | 0.6390 | headline ensemble component |
| STNDT-lite | 4-layer mask 0.5 5-seed | 5 ms | public-test | 0.3742 | not computed | 0.8949 | 0.6566 | |
| STNDT-lite | diverse_screen_n / diverse_screen_o (best train/val ensembles) | 5 ms | train/val | ~0.3751 | not computed | saved | not computed | validation screens only; not public-test |
| STNDT-lite | Screens P / Q / R1 (best candidates) | 5 ms | train/val | ~0.368–0.369 | not computed | saved | not computed | validation-negative; not promoted |
| LFADS | 100-epoch baseline (`20260603T201838Z`) | 20 ms | train/val | **0.3606** | 0.2525 | 0.8946 | 0.5818 | 100 epochs, batch 32, 20 posterior samples |
| LFADS | 50-epoch baseline (`20260603T094126Z`) | 20 ms | train/val | 0.3499 | 0.2402 | 0.8967 | 0.5843 | 50 epochs, batch 32, 20 posterior samples |
| LFADS | S2 stability, seed 0 (`20260605T060424Z`) | 5 ms | train/val | **0.3160** | 0.2121 | 0.8971 | psth skipped | 30 epochs, batch 8, lr 1e-3, grad clip 1.0, seed 0 |
| LFADS | S2 stability, seed 1 (`20260605T171044Z`) | 5 ms | train/val | 0.3154 | 0.2156 | 0.8958 | psth skipped | same config, seed 1 |
| LFADS | salvage epoch-14 export (`20260604T062750Z`) | 5 ms | train/val | 0.2902 | 0.1928 | 0.8874 | psth skipped | checkpoint epoch 14, batch 8; 20 posterior samples |

### Metric availability notes

- **STNDT-lite public-test:** co-bps, vel R², and psth R² are saved for several
  public-test runs under `results/public_test/` (see table above for key rows).
- **STNDT-lite fp-bps:** **not computed** — saved prediction HDF5s contain held-in/out
  rates only; they lack forward-prediction rate fields that `nlb_tools.evaluate`
  uses for fp-bps.
- **STNDT-lite train/val screens:** `results/benchmark_runs/*/metrics.csv` files
  generally record co-bps and vel R². fp-bps is **not computed**. psth R² is
  **not computed** because benchmark configs set `include_psth=false`.
- **LFADS 20 ms:** full metric set including psth R² (PSTH present in bundled
  20 ms HDF5).
- **LFADS 5 ms:** psth R² is **psth skipped** — the 5 ms HDF5 was built with
  `--skip-psth` after full PSTH creation caused OOM (~10 GB).

### Interpretation

- **STNDT-lite (5 ms, public-test)** remains the strongest result on co-bps
  (**0.3830**) and also reports strong vel R² (**0.9053**) and psth R² (**0.6390**).
- **LFADS 5 ms S2 (train/val)** is reproducible across two seeds (**0.3160** /
  **0.3154** co-bps) but lower on co-bps than the STNDT-lite public-test headline.
  Do not treat train/val LFADS scores as public-test equivalents.
- **LFADS 5 ms vel R²** (~**0.896–0.897**, train/val) is close to STNDT-lite
  public-test vel R² (**0.9053**), suggesting movement-related structure is
  captured even though held-out spike likelihood (co-bps) is lower on the LFADS track.
- **LFADS 20 ms (train/val)** is a useful model-family baseline (**0.3606** co-bps
  at 100 epochs) but uses different binning and evaluation contract than **5 ms
  STNDT-lite public-test** — compare only with explicit bin-size and split labels.

## Reproduce

```bash
python -m pip install -e '.[dev,neural]'
nlb-get-public-eval-data --out data/eval/eval_data_test.h5
nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_stndt_lite_depth4_mask06_5seed_public.yaml \
  --output-dir results/public_test/mc_maze_stndt_lite_depth4_mask06_5seed
nlb-evaluate-ensemble-public-test \
  --config configs/benchmarks/mc_maze_stndt_lite_diverse_lr0013_depth5_public.yaml
```
