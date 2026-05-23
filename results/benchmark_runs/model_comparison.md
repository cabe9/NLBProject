# Model Comparison

Generated from saved `metrics.csv` artifacts in `results/benchmark_runs/*/`.
The only manual part is the comparison manifest in `src/nlb_project/reporting.py`, which selects which saved run rows to display.

| model | role | history bins | rank | n_components | ridge_alpha | transform | co-bps | vel R2 | source |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| static PCA latent regression | reference | n/a | n/a | 10 | 0.1 | none | -0.0068 | 0.0756 | `results/benchmark_runs/static_pca/metrics.csv` (baseline) |
| static direct ridge | reference | n/a | n/a | n/a | 0.1 | none | 0.0017 | 0.0769 | `results/benchmark_runs/static_ridge/metrics.csv` (baseline) |
| lagged direct ridge (5 bins) | reference | 5 | n/a | n/a | 0.1 | sqrt | 0.0215 | 0.1986 | `results/benchmark_runs/lagged_ridge_single/metrics.csv` (baseline) |
| lagged reduced-rank regression (selected) | selected | 9 | 20 | n/a | 0.1 | sqrt_zscore | 0.0283 | 0.2316 | `results/benchmark_runs/lagged_rrr_sweep/metrics.csv` (improved) |
| lagged PCA latent regression (5 bins) | reference | 5 | n/a | 20 | 0.1 | sqrt_zscore | 0.0166 | 0.2400 | `results/benchmark_runs/lagged_pca_single/metrics.csv` (baseline) |
| lagged PCA latent regression (selected history) | selected | 9 | n/a | 20 | 0.1 | sqrt_zscore | 0.0266 | 0.3648 | `results/benchmark_runs/lagged_pca_history_sweep/metrics.csv` (improved) |

Takeaway:
- Short neural history is what separates the competitive models from the static ones; every lagged row beats every static row on co-bps.
- Co-bps alone does not pick a single winner among lagged models: RRR and lagged PCA finish within 0.002 co-bps of each other, but lagged PCA pulls ahead on vel R2 (0.36 vs 0.23), indicating a more behaviourally structured latent.
- Static latent dimensionality alone is not enough: static PCA sits near zero co-bps regardless of rank.

Note on numbers:
- All rows are generated under the default `log_link` rate readout. The legacy `linear` readout clipped Gaussian-ridge outputs at 1e-9 before Poisson scoring, which produced artefacts like `lagged direct ridge = -0.43 co-bps` that were pure readout pathology rather than model failure. See `results/benchmark_runs/output_head_comparison_3heads.json` for the three-head ablation (linear / log_link / poisson_glm).