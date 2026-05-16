# NLB Public Test Summary

Evaluated locally against the public NLB test target HDF5.

| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |
|---|---|---:|---:|---:|---:|---|
| baseline | mc_maze_split | 0.3121 | 0.7343 | 0.6116 | n/a | {"batch_size": 64, "d_model": 192, "device": "auto", "dropout": 0.05, "ensemble_size": 5, "heldin_loss_weight": 0.2, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.3, "max_epochs": 40, "n_heads": 4, "n_layers": 2, "neuron_embedding_scale": 0.0, "patience": 6, "seed": 0, "validation_fraction": 0.1, "weight_decay": 0.0001} |
| selected | mc_maze_split | 0.3197 | 0.7633 | 0.6251 | n/a | {"batch_size": 64, "d_model": 192, "device": "auto", "dropout": 0.05, "ensemble_size": 5, "heldin_loss_weight": 0.3, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.3, "max_epochs": 60, "n_heads": 4, "n_layers": 2, "neuron_embedding_scale": 0.0, "patience": 10, "seed": 0, "validation_fraction": 0.05, "weight_decay": 0.0001} |