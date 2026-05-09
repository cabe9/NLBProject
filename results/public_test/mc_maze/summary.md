# NLB Public Test Summary

Evaluated locally against the public NLB test target HDF5.

| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |
|---|---|---:|---:|---:|---:|---|
| baseline | mc_maze_split | 0.0199 | 0.2690 | -26.4117 | n/a | {"history_bins": 5, "input_transform": "sqrt_zscore", "log_offset": 0.001, "n_components": 20, "output_head": "log_link", "ridge_alpha": 0.1} |
| selected | mc_maze_split | 0.0268 | 0.3678 | -26.4081 | n/a | {"history_bins": 9, "input_transform": "sqrt_zscore", "log_offset": 0.001, "n_components": 20, "output_head": "log_link", "ridge_alpha": 0.1} |