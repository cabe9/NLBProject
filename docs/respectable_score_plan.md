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
| `MC_Maze 5 ms` public test | NDT-lite, tuned 192-wide 5-seed ensemble | 0.3121 | 0.7343 | 0.6116 |
| `MC_Maze 5 ms` public test | NDT-lite, stability-tuned 192-wide 5-seed ensemble | 0.3197 | 0.7633 | 0.6251 |
| `MC_Maze 5 ms` public test | NDT-lite, stability-tuned 192-wide 7-seed ensemble | 0.3229 | 0.7693 | 0.6368 |
| `MC_Maze 5 ms` public test | STNDT-lite, spatiotemporal 5-seed ensemble | 0.3302 | 0.8138 | 0.6441 |

The lagged-PCA score is useful as a reproducible floor, not as a competitive
endpoint. The STNDT-lite result is the first neural model here to move beyond
the old NDT-class public-test level: about `+0.303 co-bps` absolute over the
linear baseline. It still leaves a clear gap to stronger leaderboard methods.
On the frozen EvalAI `MC_Maze 5 ms` leaderboard, representative co-bps levels are:

| Approximate target | Public method example | co-bps |
|---|---|---:|
| NDT-class baseline | `NDT` | 0.3229 |
| top-10 neighborhood | `KubeFlow AutoLFADS` | 0.3510 |
| top-5 neighborhood | `AESMTE3 [Ensemble]` | 0.3676 |
| top score | `STNDT[Ensemble]` | 0.3862 |

EvalAI no longer accepts new submissions, so the achievement target should be
a reproducible local public-test score, not a new public leaderboard rank.

## Current neural baseline: STNDT-lite

The current headline model is a compact STNDT-inspired transformer:

- PyTorch model registered as a normal `model_type` (`stndt_lite`).
- Inputs are held-in spike counts with learned temporal embeddings.
- A temporal attention branch models population dynamics over time.
- A neuron-token branch attends across neurons and feeds spatially reweighted
  population context back into the temporal branch.
- Training uses Poisson rate prediction on held-out neurons plus masked held-in
  reconstruction; optional contrastive consistency is implemented but was not
  selected by validation in the bounded screen.
- Public-test score is produced by `nlb-evaluate-public-test`, with final fit
  on `train + val`.
- The first bounded screen selected `n_layers=3`, `d_model=192`, `dropout=0.05`,
  and `mask_prob=0.3`: single-seed train/val improved from `0.2437` to
  `0.2737`. Larger `d_model=256` and the first two contrastive settings trailed
  the selected no-contrast candidate on validation.
- Promoting the selected architecture to 5 seeds improved train/val from
  `0.2737` to `0.3094`, narrowly beating the previous NDT-lite train/val
  headline (`0.3068`). The single justified public-test run improved from
  `0.3229` to `0.3302`.

This is a useful score improvement and a better modeling story, not a
leaderboard-equivalent STNDT reproduction. It remains `0.0560 co-bps` below the
frozen `STNDT[Ensemble]` public-test target (`0.3862`).

## Axial STNDT guardrail

The repo also includes a more explicit neuron-identity architecture:

- PyTorch model registered as `model_type: stndt_axial`.
- Uses held-in and held-out neuron embeddings, per-time spatial latent
  cross-attention over held-in neurons, temporal population attention, a
  learned mask token for held-in reconstruction, and held-out neuron query
  decoding.
- A direct all-pairs neuron-attention version was too slow for the bounded
  screen, so the committed version uses spatial latents to make the probe
  tractable.
- The first bounded mc_maze probe did **not** justify promotion: CV mean was
  `-0.0024 co-bps`, and the completed train/val artifact in
  `results/benchmark_runs/stndt_axial_screen/metrics.csv` scored `-0.3446`
  selected co-bps. Do not ensemble it or run public-test without a new
  train/val result that beats the current `0.3094` STNDT-lite gate.

## Previous neural baseline: NDT-lite

The repo also includes a small Neural Data Transformer-style model:

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
- A curated NDT-lite architecture screen selected `d_model=192`, `dropout=0.05`,
  and `mask_prob=0.3`: single-seed train/val improved from `0.2481` to `0.2523`.
  Promoting that config to 5 seeds improved train/val from `0.2713` to `0.2891`,
  and the single public-test run improved from `0.3004` to `0.3121`.
- A 7-seed follow-up for the tuned 192-wide model improved train/val only to
  `0.2915`, below the stricter public-test gate, so it was not scored on public
  test.
- A stability pass around the 192-wide model found that the useful knobs were
  not bigger depth/heads or smaller batches. The single-seed full train/val
  probe improved from `0.2523` to `0.2686` by using
  `heldin_loss_weight=0.3`, `validation_fraction=0.05`, and
  `max_epochs=60`/`patience=10`.
- Promoting that stability config to 5 seeds improved train/val from `0.2891`
  to `0.3033`, clearing the public-test gate. The public-test score improved
  from `0.3121` to `0.3197`.
- A validation-only ensemble-size follow-up then selected 7 seeds over 5:
  train/val improved from `0.3033` to `0.3068`, and the justified public-test
  run improved from `0.3197` to `0.3229`.
- A first factorized neuron/time transformer is implemented as
  `model_type: ndt_factorized`, but the bounded mc_maze train/val run did
  **not** justify promotion: CV mean was `0.0153 co-bps`, and the completed
  train/val artifact in `results/benchmark_runs/ndt_factorized_sweep/metrics.csv`
  scored below the current NDT-lite reference. Treat it as an architecture
  foundation/guardrail, not a score-improvement path.

The first practical target, `>0.32 co-bps`, is now met locally without tuning
directly on public-test targets. The next target, `0.3862`, should be framed as
closing the gap to the frozen STNDT ensemble through better modeling, not by
claiming a live EvalAI rank. Do not spend more public-test runs on the first
depth/dropout/cosine schedule sweep, the simple event-embedding sweep, the
first factorized architecture, the first axial STNDT probe, contrastive
settings from the first STNDT-lite screen, or larger ensembles unless a later
train/val run beats the current `0.3094` STNDT-lite train/val result by a
meaningful margin.

Run the neural score configs with:

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
  --config configs/benchmarks/mc_maze_ndt_lite_arch_screen.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_arch_5seed_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_arch_5seed_sweep.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_192_5seed

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_192_ensemble_sweep.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_192_stability_screen.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_192_stability_5seed_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_192_stability_5seed_sweep.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_192_stability_5seed

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_lite_192_stability_ensemble_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_ndt_lite_192_stability_ensemble_sweep.yaml \
  --output-dir results/public_test/mc_maze_ndt_lite_192_stability_7seed

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_ndt_factorized_sweep.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_stndt_lite_screen.yaml

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_stndt_lite_ensemble_sweep.yaml

nlb-evaluate-public-test \
  --config configs/benchmarks/mc_maze_stndt_lite_ensemble_sweep.yaml \
  --output-dir results/public_test/mc_maze_stndt_lite_5seed

nlb-run-experiment \
  --config configs/benchmarks/mc_maze_stndt_axial_screen.yaml
```

### Bounded experiment brief (tooling / lighter models)

Use this shape when delegating **config-only** or **small scripted** changes so scope stays reviewable:

1. **Hypothesis** — One sentence (e.g. “larger `d_model` improves co-bps before ensemble”).
2. **Allowed edits** — List paths (e.g. `configs/benchmarks/mc_maze_ndt_lite.yaml` only, or `improvement` section only).
3. **Caps** — Explicit bounds (e.g. `max_epochs ≤ 80`, `d_model_grid ⊆ {64, 128}`, do not change `baseline` defaults).
4. **Verification** — `ruff check .`, `ruff format --check .`, `mypy src scripts tests`, `pytest -q`; treat **co-bps / R² numbers in docs** as authoritative only after you rerun `nlb-evaluate-public-test` locally.

Reserve architecture choices (causal attention, schedules, new objectives) and multi-axis science sweeps for human judgment or a stronger model pass.

## Why not only scale ensembles?

The current best model is now a compact STNDT-inspired transformer, not the
older lagged-PCA baseline. Pure ensemble scaling helped only after validation
supported it, and the STNDT-lite 5-seed run already used that gate. Past this
point, larger ensembles are likely to be a poor tradeoff unless train/val
improves clearly first. The first axial neuron-identity probe was the right
idea to test, but its validation result says the current implementation is not
the path to `0.3862`. The next useful work should be narrower: inspect why
STNDT-lite succeeds while the axial decoder collapses, then try one fix that
preserves the STNDT-lite temporal backbone instead of replacing the whole
readout stack. SSM/Mamba-style temporal mixers are worth considering later, but
the `mc_maze` sequences are short enough that the current gap is unlikely to be
mostly a long-context efficiency problem.

## References

- NLB local test data note: https://github.com/neurallatents/nlb_tools
- Frozen leaderboard page/API: https://eval.ai/web/challenges/challenge-page/1256/leaderboard
- Neural Data Transformers: https://github.com/snel-repo/neural-data-transformers
- STNDT: https://github.com/trungle93/STNDT
- S5: https://github.com/lindermanlab/S5
