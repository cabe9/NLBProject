# Model Comparison

Generated from saved `metrics.csv` artifacts in `results/benchmark_runs/*/`.
The only manual part is the comparison manifest in `src/nlb_project/reporting.py`, which selects which saved run rows to display.

| model | role | history bins | rank | n_components | ridge_alpha | transform | co-bps | vel R2 | source |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| static PCA latent regression | reference | n/a | n/a | 10 | 0.1 | none | 0.0039 | 0.0755 | `results/benchmark_runs/static_pca/metrics.csv` (baseline) |
| static direct ridge | reference | n/a | n/a | n/a | 0.1 | none | -0.0335 | 0.0761 | `results/benchmark_runs/static_ridge/metrics.csv` (baseline) |
| lagged direct ridge (5 bins) | reference | 5 | n/a | n/a | 0.1 | sqrt | -0.4301 | 0.2007 | `results/benchmark_runs/lagged_ridge_single/metrics.csv` (baseline) |
| lagged reduced-rank regression (selected) | selected | 5 | 5 | n/a | 0.1 | sqrt_zscore | -0.0091 | 0.1594 | `results/benchmark_runs/lagged_rrr_sweep/metrics.csv` (improved) |
| lagged PCA latent regression (5 bins) | reference | 5 | n/a | 20 | 0.1 | sqrt_zscore | 0.0418 | 0.2441 | `results/benchmark_runs/lagged_pca_single/metrics.csv` (baseline) |
| lagged PCA latent regression (selected history) | selected | 9 | n/a | 20 | 0.1 | sqrt_zscore | 0.0486 | 0.3730 | `results/benchmark_runs/lagged_pca_history_sweep/metrics.csv` (improved) |

Takeaway:
- The winning change was adding short neural history and compressing that history before regression.
- Temporal context mattered more than static latent dimensionality alone.
- A supervised reduced-rank mapping did not recover the same co-smoothing gain as lagged PCA.