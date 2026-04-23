# Postmortem: the `linear` output head bug

This is a short case study of a real mistake made earlier in this project and how it was diagnosed. The numbers in this repo's current headline are scored correctly; the ones referenced below are preserved in `results/benchmark_runs/output_head_comparison_3heads.json` for reference only.

## The finding (as originally reported)

An earlier draft of this repo reported a large gap between lagged PCA and lagged direct ridge on `mc_maze`:

- `lagged_pca_latent_regression` → `co-bps = +0.049`
- `lagged_ridge_direct` → `co-bps = −0.43`

The original writeup concluded that a low-rank PCA bottleneck was necessary to get competitive `co-bps` on lagged features, and that supervised reduced-rank regression failed to recover the gain.

That conclusion was wrong.

## The actual cause

Every regression model in the repo used a single shared "output head" to convert its raw outputs to non-negative neuron rates before Poisson scoring. The original head (`linear`) fit a Gaussian ridge on raw spike counts and clipped the predictions to `[1e-9, 1e20]` at inference:

```python
rates = np.clip(coef @ X + intercept, 1e-9, 1e20)
```

For models that produced many near-zero or slightly negative predictions — which `lagged_ridge_direct` did, because its feature space was high-dimensional and under-regularised relative to `lagged_pca_latent_regression` — this clip floor dominated the log-likelihood. Poisson `bits/spike` against a `1e-9` predicted rate is a very large negative number. The `−0.43 co-bps` was almost entirely clip-floor noise, not model failure.

## The fix

The shared readout was replaced by a pluggable `output_head` module (`src/nlb_project/models/output_head.py`) with three options:

- **`log_link`** (new default): ridge is fit on `log(count + log_offset)`. Predictions are exponentiated at inference, guaranteeing strictly positive rates without a clip floor. This is the `co-bps`-correct head.
- **`poisson_glm`**: a Poisson GLM with a log link fit via IRLS. Slower, but the "textbook" likelihood-matched choice.
- **`linear`**: the legacy clipped Gaussian head, kept for ablations only.

All runs now record the effective head in `run_metadata.json`, and the headline comparison in `results/benchmark_runs/model_comparison.csv` is scored under `log_link`.

## Result after the fix

Under the corrected `log_link` head:

| model | old `co-bps` (`linear`) | new `co-bps` (`log_link`) |
|---|---:|---:|
| lagged direct ridge (5 bins) | −0.43 | +0.0215 |
| lagged reduced-rank regression (selected) | — | +0.0283 |
| lagged PCA (selected history) | +0.049 | +0.0266 |

The real story is narrower and more interesting than the old one:

- The gap between lagged direct ridge and lagged PCA on `co-bps` is ~0.005–0.01, not 0.5.
- RRR and lagged PCA tie within `~0.002 co-bps`; the remaining discriminator is `vel R²`, where lagged PCA's unsupervised bottleneck captures more behaviourally-aligned structure.
- The static-vs-lagged gap (temporal context) is still real and still decisive.

The three-head ablation used to confirm this is checked in at `results/benchmark_runs/output_head_comparison_3heads.json`.

## Lessons

1. **Shared infrastructure across models is load-bearing.** A single clip floor silently re-wrote the story for an entire family of models. Any post-processing that applies identically to "baseline" and "improvement" is a candidate for hidden bias.
2. **A suspicious win deserves a suspicious check.** `+0.049` vs `−0.43` was a red flag that no pair of sensibly-specified linear models should produce. The gap should have triggered an ablation before it triggered a narrative.
3. **The readout is a model choice.** Ridge-on-log-counts (`log_link`), a Poisson GLM, and clipped-Gaussian are three different models of the rate, not three different implementations of the same model.
4. **Keep the bad artifact.** `output_head_comparison_3heads.json` is deliberately tracked in the repo — being able to reproduce the old wrong number is part of trusting the new right one.
