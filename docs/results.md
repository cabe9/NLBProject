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
