# Path to a respectable NLB score

This repo now has a local public-test harness for the frozen NLB'21 leaderboard:

```bash
make public-eval-data
make public-test
```

The current reproducible public-test scores:

| dataset split | model | co-bps | vel R² | psth R² |
|---|---|---:|---:|---:|
| `MC_Maze 5 ms` public test | lagged PCA latent regression, selected history | 0.0268 | 0.3678 | −26.4081 |
| `MC_Maze 5 ms` public test | NDT-lite temporal transformer | 0.2338 | 0.6412 | 0.3397 |
| `MC_Maze 5 ms` public test | NDT-lite, 3-seed ensemble | 0.2481 | 0.6589 | 0.4086 |
| `MC_Maze 5 ms` public test | NDT-lite, wider 3-seed ensemble | 0.2951 | 0.7096 | 0.5498 |
| `MC_Maze 5 ms` public test | NDT-lite, wider 5-seed ensemble | 0.3004 | 0.7222 | 0.5601 |

The lagged-PCA score is useful as a reproducible floor, not as a competitive
endpoint. The wider NDT-lite ensemble result is the first clearly respectable
neural-sequence baseline: about `+0.274 co-bps` absolute over the linear
baseline, while still leaving a clear gap to old public leaderboard methods.
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
- Optional seed ensembling averages independently initialized NDT-lite models,
  matching a common pattern in high-scoring leaderboard submissions.
- A validation-led width sweep selected `d_model=128` over the initial
  `d_model=64`, improving train/val co-bps from `0.1934` to `0.2481` before
  public-test scoring.
- A first bounded depth/dropout/mask/scheduler sweep did **not** improve the
  final train/val result: the selected no-dropout model scored `0.2320 co-bps`
  versus the `0.2481` width reference, so it should not be promoted to
  public-test scoring.
- A first neuron-event embedding sweep also did **not** improve validation:
  positive scales `[0.1, 0.25, 0.5]` all trailed the `0.0` scale anchor, so the
  simple event-embedding path should remain off for the current headline.
- A validation-led ensemble-size sweep selected `ensemble_size=5` over the
  existing 3-seed width ensemble: train/val co-bps improved from `0.2674` to
  `0.2713`, and the single public-test run improved from `0.2951` to `0.3004`.
- A first factorized neuron/time transformer is implemented as
  `model_type: ndt_factorized`, but the bounded mc_maze train/val run did
  **not** justify promotion: CV mean was `0.0153 co-bps`, and the completed
  train/val artifact in `results/benchmark_runs/ndt_factorized_sweep/metrics.csv`
  scored below the current NDT-lite reference. Treat it as an architecture
  foundation/guardrail, not a score-improvement path.

The next practical target is `>0.32 co-bps`, roughly old NDT-class performance,
without tuning directly on the public test result. Likely useful changes are
better validation stability around the current NDT-lite model, then a more
faithful NDT/STNDT-style architecture if stability work plateaus. Do not spend
more public-test runs on the first depth/dropout/cosine schedule sweep, the
simple event-embedding sweep, the first factorized architecture, or larger
ensembles unless a later train/val run beats the current `0.2713` 5-seed
train/val result.

Run the initial NDT-lite config with:

```bash
python -m pip install -e '.[dev,neural]'
nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_ensemble.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_ensemble

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_width_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_width_ensemble.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_width_ensemble

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_tuning_sweep.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_neuron_sweep.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_ensemble_size_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_ensemble_size_sweep.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_5seed_ensemble

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_factorized_sweep.yaml
```

### Bounded experiment brief (tooling / lighter models)

Use this shape when delegating **config-only** or **small scripted** changes so scope stays reviewable:

1. **Hypothesis** — One sentence (e.g. “larger `d_model` improves co-bps before ensemble”).
2. **Allowed edits** — List paths (e.g. `configs/benchmarks/mc_maze_ndt_lite.yaml` only, or `improvement` section only).
3. **Caps** — Explicit bounds (e.g. `max_epochs ≤ 80`, `d_model_grid ⊆ {64, 128}`, do not change `baseline` defaults).
4. **Verification** — `ruff check .`, `ruff format --check .`, `mypy src scripts tests`, `pytest -q`; treat **co-bps / R² numbers in docs** as authoritative only after you rerun `nlb-evaluate-public-test` locally.

Reserve architecture choices (causal attention, schedules, new objectives) and multi-axis science sweeps for human judgment or a stronger model pass.

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
