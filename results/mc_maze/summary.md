# NLB MC_Maze Summary

Best validated model family: `lagged_pca_latent_regression`

| run | co-bps | vel R2 | psth R2 | params |
|---|---:|---:|---:|---|
| reference | 0.0418 | 0.2441 | -24.0829 | {"history_bins": 5, "input_transform": "sqrt_zscore", "n_components": 20, "ridge_alpha": 0.1} |
| selected | 0.0486 | 0.3730 | -24.0626 | {"history_bins": 9, "input_transform": "sqrt_zscore", "n_components": 20, "ridge_alpha": 0.1} |

Interpretation:
- Temporal context plus latent compression materially improved co-smoothing over the original static PCA setup.
- Delta co-bps (selected - reference): **0.0068**