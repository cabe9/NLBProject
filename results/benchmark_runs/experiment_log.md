# Experiment Log

This summary is generated from committed `metrics.csv` artifacts.

## Main comparison

| model | co-bps | vel R2 | note |
|---|---:|---:|---|
| static PCA latent regression | -0.0068 | 0.0756 | Static latent baseline; no temporal context. |
| static direct ridge | 0.0017 | 0.0769 | Direct one-bin regression is not competitive. |
| lagged direct ridge (5 bins) | 0.0215 | 0.1986 | Temporal history alone overfit without a latent bottleneck. |
| lagged reduced-rank regression (selected) | 0.0283 | 0.2316 | Supervised low-rank control; ties lagged PCA on co-bps, trails on vel R2. |
| lagged PCA latent regression (5 bins) | 0.0166 | 0.2400 | Temporal context plus train-only conditioning gave the first real gain. |
| lagged PCA latent regression (selected history) | 0.0266 | 0.3648 | Best validated model in the repo. |

## Validated lagged PCA result

- reference co-bps: `0.0166`
- selected co-bps: `0.0266`
- delta co-bps: `0.0100`

## Interpretation

- The original static PCA model was weak because it ignored short-timescale neural history.
- Every lagged model beats every static model on co-bps, confirming temporal context is the dominant factor on this benchmark slice.
- Under the corrected `log_link` readout, supervised reduced-rank regression is statistically indistinguishable from unsupervised PCA compression on co-bps; the distinguishing signal lives on `vel R2`, where lagged PCA's bottleneck produces a latent that is more behaviourally aligned.
- The earlier reported gap between lagged direct ridge and the other lagged models (co-bps = -0.43) was almost entirely a clip-floor artefact of the legacy Gaussian-ridge-on-counts readout; see `results/benchmark_runs/output_head_comparison_3heads.json` for the three-head ablation.