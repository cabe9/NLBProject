# Experiment Log

This summary is generated from committed `metrics.csv` artifacts.

## Main comparison

| model | co-bps | vel R2 | note |
|---|---:|---:|---|
| static PCA latent regression | 0.0039 | 0.0755 | Static latent baseline; no temporal context. |
| static direct ridge | -0.0335 | 0.0761 | Direct one-bin regression is not competitive. |
| lagged direct ridge (5 bins) | -0.4301 | 0.2007 | Temporal history alone overfit without a latent bottleneck. |
| lagged reduced-rank regression (selected) | -0.0091 | 0.1594 | A supervised low-rank mapping stayed worse than lagged PCA on co-bps. |
| lagged PCA latent regression (5 bins) | 0.0418 | 0.2441 | Temporal context plus train-only conditioning gave the first real gain. |
| lagged PCA latent regression (selected history) | 0.0486 | 0.3730 | Best validated model in the repo. |

## Validated lagged PCA result

- reference co-bps: `0.0418`
- selected co-bps: `0.0486`
- delta co-bps: `0.0068`

## Interpretation

- The original static PCA model was weak because it ignored short-timescale neural history.
- Direct lagged ridge showed that temporal context alone is not enough; the lagged design needs compression.
- Lagged PCA latent regression was the first model family that improved co-smoothing substantially while remaining simple and interpretable.
- Lagged reduced-rank regression did not beat lagged PCA, which suggests the PCA bottleneck was already a better fit than this simple supervised low-rank mapping.