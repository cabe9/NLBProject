# NLB Public Test Summary

Evaluated locally against the public NLB test target HDF5.

| run | split | co-bps | vel R2 | psth R2 | fp-bps | params |
|---|---|---:|---:|---:|---:|---|
| baseline | mc_maze_split | 0.2951 | 0.7096 | 0.5498 | n/a | {"batch_size": 64, "d_model": 128, "device": "auto", "dropout": 0.1, "ensemble_size": 3, "heldin_loss_weight": 0.2, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.2, "max_epochs": 40, "n_heads": 4, "n_layers": 2, "neuron_embedding_scale": 0.0, "patience": 6, "seed": 0, "validation_fraction": 0.1, "weight_decay": 0.0001} |
| selected | mc_maze_split | 0.3004 | 0.7222 | 0.5601 | n/a | {"batch_size": 64, "d_model": 128, "device": "auto", "dropout": 0.1, "ensemble_size": 5, "heldin_loss_weight": 0.2, "input_transform": "sqrt_zscore", "learning_rate": 0.001, "lr_schedule": "constant", "mask_prob": 0.2, "max_epochs": 40, "n_heads": 4, "n_layers": 2, "neuron_embedding_scale": 0.0, "patience": 6, "seed": 0, "validation_fraction": 0.1, "weight_decay": 0.0001} |