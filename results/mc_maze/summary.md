# NLB MC_Maze Summary

Model family: `lagged_pca_latent_regression`

| run | co-bps | vel R2 | psth R2 | params |
|---|---:|---:|---:|---|
| reference | 0.0166 | 0.2400 | -24.1551 | {"history_bins": 5, "input_transform": "sqrt_zscore", "log_offset": 0.001, "n_components": 20, "output_head": "log_link", "ridge_alpha": 0.1} |
| selected | 0.0266 | 0.3648 | -24.1502 | {"history_bins": 9, "input_transform": "sqrt_zscore", "log_offset": 0.001, "n_components": 20, "output_head": "log_link", "ridge_alpha": 0.1} |

Delta co-bps (selected - reference): **0.0100**