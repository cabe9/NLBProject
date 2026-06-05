# NLB MC_Maze — Project Summary

Reproducible modeling pipeline for the Neural Latents Benchmark (NLB'21) **MC_Maze**
dataset: predict neural population activity under a standard held-out evaluation protocol,
with validation-gated iteration and tracked benchmark artifacts.

## Problem and metric

**Goal:** Model motor-cortex population dynamics during a maze-reaching task from
recorded neural activity.

**Core task:** From **held-in** neuron spike counts (and related inputs), predict
**held-out** neuron spike activity on the same trials and time bins. Models are fit on
train data and scored on held-out validation splits before any public-test evaluation.

**Primary metric:** **Co-smoothing bits per spike (co-bps)** — the NLB co-bps score
from the official `nlb_tools` evaluator (higher is better). Auxiliary metrics include
velocity R² and PSTH R² where applicable.

**Bin size:** The STNDT-lite headline uses **5 ms** binned spikes. Scores reported here
are **local public-test evaluations** against the frozen NLB target HDF5 (SHA-verified
download), not live EvalAI leaderboard ranks.

## Main result

The best reproducible result in this repository is a **validation-selected mixed
STNDT-lite ensemble**:

- 4-layer model with `learning_rate=0.0013` (5 seeds, prediction-averaged)
- 5-layer model with `dropout=0.08` (5 seeds, prediction-averaged)

Both members share the validated masked-reconstruction objective (e.g. `mask_prob=0.6`,
full held-in loss weight, mask token, temporal neuron-event identity, spike-weighted
masked loss).

| Result | co-bps (local public-test) |
|--------|---------------------------:|
| **Mixed STNDT-lite ensemble (headline)** | **0.3830** |
| Prior 4-layer mask-0.6 5-seed ensemble | 0.3795 |
| Frozen MC_Maze 5 ms rank-1 reference (`STNDT[Ensemble]`) | 0.3862 |

The headline improves over earlier in-repo STNDT-lite checkpoints and classical baselines
(e.g. lagged PCA latent regression at ~0.027 co-bps on the same evaluation setup). It
remains slightly below the historical frozen reference; closing that gap would require
new structure or favorable validation→test transfer, not repeated near-variant probing.

Reproduce the headline public-test evaluation:

```bash
python -m pip install -e '.[dev,neural]'
nlb-get-public-eval-data --out data/eval/eval_data_test.h5
nlb-evaluate-ensemble-public-test \
  --config configs/benchmarks/mc_maze_stndt_lite_diverse_lr0013_depth5_public.yaml
```

## Model approach (STNDT-lite)

**STNDT-lite** is a compact spatiotemporal transformer: temporal attention over population
dynamics, a neuron-token branch, and masked Poisson reconstruction on held-in spikes to
train rates for held-in and held-out neurons. The repo registers multiple model families
(linear baselines through neural transformers) behind a single `fit_predict_*` contract
and declarative YAML configs.

Progression that mattered: NDT-lite → STNDT-lite identity/spike-weighted objectives →
Screen C CD-reconcile winner → 4-layer mask-0.6 ensemble → diverse two-member mixed
ensemble above.

## Validation-negative screens (stopped paths)

Late-stage **knob screens on the same backbone** were run validation-only (train/val
gates; no public-test promotion). Committed configs document intent and stop rules:

| Screen | Hypothesis | Outcome |
|--------|------------|---------|
| **P** — full capacity / long train | Wider/deeper STNDT-lite with cosine+warmup and extended epochs | **Validation-negative** — did not clear promotion gates; headline unchanged |
| **Q** — block/span masking | Contiguous time-span masking vs default Bernoulli masking | **Validation-negative** — block variants underperformed Bernoulli control on train/val |
| **R1** — unit affine calibration | Per-unit held-out logit scale/bias after training | **Validation-negative** — calibrated variants did not beat the uncalibrated control by the required train/val margin |

These outcomes motivated **stopping further STNDT-lite tweak screens** on this stack.

## Diagnostics (headline model, validation split)

Committed tooling analyzes validation predictions without retraining:

- **`scripts/export_headline_val_predictions.py`** — export val rates for the headline ensemble
- **`scripts/diagnose_validation_residuals.py`** — slice/unit residual and go/no-go for targeted loss reweighting
- **`scripts/diagnose_calibration_dispersion.py`** — calibration curves and variance-vs-mean (dispersion) checks
- **`configs/diagnostics/mc_maze_headline_val_predictions.yaml`** — diagnostic config

Findings (validation split, headline ensemble):

- **Residuals:** Errors were **diffuse across movement phase and speed slices** — no strong concentrated slice or unit signal to justify phase/speed loss reweighting (Screen R loss path: no-go).
- **Dispersion:** Variance/mean ratios were **approximately Poisson** (~1.0); no case for a first-pass overdispersion readout change from dispersion alone.
- **Calibration:** Some **unit-level miscalibration** appeared in diagnostics, but **Screen R1 affine calibration did not improve train/val co-bps** — calibration signal did not translate into validation gain.

## Engineering discipline

- **Train/validation gating** for model selection and screen promotion; explicit gates in screen YAMLs.
- **No public-test promotion** without validation evidence and deliberate approval — public-test uses train+val refit against the frozen target schema.
- **Reproducibility:** versioned configs under `configs/benchmarks/`, pytest coverage for configs and STNDT-lite hooks, CLI entry points (`nlb-run-experiment`, `nlb-evaluate-public-test`, ensemble evaluator), run metadata and whitelisted `metrics.csv` artifacts under `results/benchmark_runs/`.
- **CI** (ruff, mypy, pytest) on the core pipeline; optional PyTorch install for neural families.

## Expansion: LFADS / AutoLFADS baseline (separate track)

A parallel branch (`lfads-baseline-setup`, pushed separately) adds an **LFADS
(`lfads-torch`) smoke train, rate export, and NLB evaluation bridge** — a different
model family, not another STNDT-lite screen.

- LFADS upstream MC_Maze tooling targets **20 ms** binning; the STNDT-lite headline is **5 ms**.
- **Do not compare 20 ms and 5 ms co-bps without explicit bin-size labels.**
- LFADS paths (`external/lfads-torch/`, `results/lfads_smoke/`, local envs) are gitignored on the STNDT branch; smoke artifacts stay local.

## Conclusion

This project delivers a **reproducible, validation-disciplined neural baseline** for NLB
MC_Maze at **0.3830 co-bps** (local public-test), with documented negative screens and
diagnostics explaining why late STNDT-lite knobs were abandoned.

**Stopped:** Local STNDT-lite hyperparameter and objective screens on the current
backbone — validation evidence no longer supported further promotion.

**Next valid direction:** A **different model-family baseline** (e.g. LFADS at the
correct bin size and data contract), not additional small STNDT-lite tweaks without new
structural signal.

## References in this repo

- [`README.md`](../README.md) — quick start and headline commands
- [`docs/results.md`](results.md) — score table and reproduction
- [`docs/models.md`](models.md) — model families and design principles
- [`docs/architecture.md`](architecture.md) — pipeline layout
