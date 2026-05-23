# Results

This repo evaluates NLB'21 `mc_maze` locally against the public test-target
HDF5 from the official `nlb_tools` codepack. EvalAI submissions are closed, so
these scores are reproducible local public-test scores, not live leaderboard
ranks.

## Headline

The current best model is `stndt_lite` with the Screen C CD-reconcile winner:
higher mask probability, full held-in loss weight, mask token, temporal
neuron-event identity, spike-weighted masked reconstruction, and a
validation-selected 5-seed ensemble.

| Model | co-bps | vel R² | psth R² |
|---|---:|---:|---:|
| lagged PCA latent regression, selected history | 0.0268 | 0.3678 | -26.4081 |
| NDT-lite, stability-tuned 192-wide 7-seed ensemble | 0.3229 | 0.7693 | 0.6368 |
| STNDT-lite, spatiotemporal 5-seed ensemble | 0.3302 | 0.8138 | 0.6441 |
| STNDT-lite, temporal identity + spike-weighted 5-seed ensemble | 0.3413 | 0.8451 | -1.8622 |
| **STNDT-lite, Screen C CD-reconcile winner 5-seed ensemble** | **0.3649** | **0.8911** | **0.6548** |

## Context

The lagged PCA baseline is a reproducible floor. The current neural result is
about `+0.338 co-bps` above that baseline and above the prior STNDT-lite
identity+spike public-test level (`0.3413 co-bps`).

The frozen `MC_Maze 5 ms` rank-1 leaderboard reference is `0.3862 co-bps`
(`STNDT[Ensemble]`). This repo's current `0.3649 co-bps` result is a meaningful
local benchmark improvement, but it should not be described as a live
leaderboard rank.

## Reproduce

```bash
python -m pip install -e '.[dev,neural]'
nlb-get-public-eval-data --out data/eval/eval_data_test.h5
nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_stndt_lite_cd_reconcile_winner_5seed_public.yaml \
  --output-dir results/public_test/mc_maze_stndt_lite_cd_reconcile_winner_5seed
```
