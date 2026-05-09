# Path to a respectable NLB score

This repo now has a local public-test harness for the frozen NLB'21 leaderboard:

```bash
make public-eval-data
make public-test
```

The current selected lagged-PCA baseline scores:

| dataset split | model | co-bps | vel R2 |
|---|---|---:|---:|
| `MC_Maze 5 ms` public test | lagged PCA latent regression, selected history | 0.0268 | 0.3678 |

That number is useful as a reproducible floor, not as a competitive endpoint.
On the frozen EvalAI `MC_Maze 5 ms` leaderboard, representative co-bps levels are:

| Approximate target | Public method example | co-bps |
|---|---|---:|
| NDT-class baseline | `NDT` | 0.3229 |
| top-10 neighborhood | `KubeFlow AutoLFADS` | 0.3510 |
| top-5 neighborhood | `AESMTE3 [Ensemble]` | 0.3676 |
| top score | `STNDT[Ensemble]` | 0.3862 |

EvalAI no longer accepts new submissions, so the achievement target should be
a reproducible local public-test score, not a new public leaderboard rank.

## Next implementation target: NDT-lite

The next useful model is a small Neural Data Transformer-style model:

- PyTorch model registered as a normal `model_type`.
- Inputs are held-in spike counts plus optional time-bin and neuron embeddings.
- Training objective is masked Poisson negative log likelihood on observed
  train-side neurons.
- Validation score is selected through the existing train/val machinery.
- Public-test score is produced by `make public-test`, with final fit on
  `train + val`.

First target: exceed `0.25 co-bps` on `MC_Maze 5 ms` public test. That would
show the repo has crossed from linear baselines into neural sequence-model
territory. After that, a serious target is `>0.32 co-bps`, roughly old NDT
class performance.

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
