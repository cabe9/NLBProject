# NLB Public Test Summary

Evaluated locally against the public NLB test target HDF5.

| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |
|---|---|---:|---:|---:|---:|---|
| baseline | mc_maze_split | 0.2800 | 0.7550 | 0.5321 | n/a | {"batch_size": 64, "contrast_loss_weight": 0.0, "contrast_mask_prob": 0.05, "contrast_temperature": 0.07, "d_model": 192, "device": "auto", "dropout": 0.05, "ensemble_size": 1, "heldin_loss_weight": 0.3, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.3, "max_epochs": 60, "n_heads": 4, "n_layers": 3, "patience": 10, "seed": 0, "spatial_n_heads": 4, "validation_fraction": 0.05, "weight_decay": 0.0001} |
| selected | mc_maze_split | 0.3302 | 0.8138 | 0.6441 | n/a | {"batch_size": 64, "contrast_loss_weight": 0.0, "contrast_mask_prob": 0.05, "contrast_temperature": 0.07, "d_model": 192, "device": "auto", "dropout": 0.05, "ensemble_size": 5, "heldin_loss_weight": 0.3, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.3, "max_epochs": 60, "n_heads": 4, "n_layers": 3, "patience": 10, "seed": 0, "spatial_n_heads": 4, "validation_fraction": 0.05, "weight_decay": 0.0001} |