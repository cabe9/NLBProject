# Path to a respectable NLB score

This repo now has a local public-test harness for the frozen NLB'21 leaderboard:

```bash
make public-eval-data
make public-test
```

The current reproducible public-test scores:

| dataset split | model | co-bps | vel R2 |
|---|---|---:|---:|
| `MC_Maze 5 ms` public test | lagged PCA latent regression, selected history | 0.0268 | 0.3678 |
| `MC_Maze 5 ms` public test | NDT-lite temporal transformer | 0.2338 | 0.6412 |

The lagged-PCA score is useful as a reproducible floor, not as a competitive
endpoint. The NDT-lite result is the first measurable neural-sequence baseline:
about `+0.207 co-bps` absolute over the linear baseline, while still leaving a
clear gap to old public leaderboard methods.
On the frozen EvalAI `MC_Maze 5 ms` leaderboard, representative co-bps levels are:

| Approximate target | Public method example | co-bps |
|---|---|---:|
| NDT-class baseline | `NDT` | 0.3229 |
| top-10 neighborhood | `KubeFlow AutoLFADS` | 0.3510 |
| top-5 neighborhood | `AESMTE3 [Ensemble]` | 0.3676 |
| top score | `STNDT[Ensemble]` | 0.3862 |

EvalAI no longer accepts new submissions, so the achievement target should be
a reproducible local public-test score, not a new public leaderboard rank.

## Current neural baseline: NDT-lite

The repo now includes a small Neural Data Transformer-style model:

- PyTorch model registered as a normal `model_type` (`ndt_lite`).
- Inputs are held-in spike counts with learned temporal embeddings.
- Training objective is masked Poisson negative log likelihood on observed
  train-side neurons.
- Validation score is selected through the existing train/val machinery.
- Public-test score is produced by `nlb-evaluate-public-test`, with final fit
  on `train + val`.

The next practical target is to push this single-model NDT-lite run above
`0.25 co-bps` without tuning directly on the public test result. After that, a
serious target is `>0.32 co-bps`, roughly old NDT-class performance. Likely
useful changes are wider/deeper transformer sweeps, mask-ratio and dropout
sweeps, a stronger learning-rate schedule, and averaging a small seed ensemble.

Run the initial NDT-lite config with:

```bash
python -m pip install -e '.[dev,neural]'
nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite
```

## Why not keep tuning the current model?

The current best model is linear after lagged PCA features. It is valuable as
a transparent baseline, but the leaderboard methods that score well are
sequence models: NDT/STNDT-style transformers, LFADS-style recurrent latent
models, S5/state-space models, and ensembles. Hyperparameter tuning the
linear path is unlikely to close a `~0.30 co-bps` gap.

## References

- NLB local test data note: https://github.com/neurallatents/nlb_tools
- Frozen leaderboard page/API: https://eval.ai/web/challenges/challenge-page/1256/leaderboard
- Neural Data Transformers: https://github.com/snel-repo/neural-data-transformers
- STNDT: https://github.com/trungle93/STNDT
- S5: https://github.com/lindermanlab/S5
