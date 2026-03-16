# Neural Latents Benchmark (NLB'21) Project

This repository is a small, reproducible NLB'21 project focused on `mc_maze`.

It includes:
1. an end-to-end local evaluation pipeline built on official `nlb_tools`
2. multiple lightweight model options that can run on a laptop
3. reproducible configs, saved result tables, and local tests

Currently implemented model families:
- `smoothing`
- `pca_latent_regression`
- `ridge_direct`

Primary metric: `co-bps`. Secondary metrics: `vel R2` and `psth R2` when available.

## What the code does

The pipeline:
1. loads NWB data with `nlb_tools.nwb_interface.NWBDataset`
2. builds train/eval/target tensors with official `nlb_tools.make_tensors`
3. runs a reference parameter set
4. runs cross-validation to choose a selected parameter set for the active model family
5. writes prediction `.h5` artifacts and local evaluation metrics

The checked-in config currently targets:

```yaml
model_type: pca_latent_regression
dataset_name: mc_maze
```

## Repository layout

- `src/nlb_project/`: package code
- `src/nlb_project/pipeline.py`: experiment orchestration
- `src/nlb_project/models/`: model implementations
- `scripts/run_experiment.py`: CLI entrypoint
- `scripts/get_data.py`: DANDI download helper
- `configs/mc_maze_smoothing.yaml`: current experiment config
- `results/mc_maze/`: saved metrics and summary tables
- `tests/`: unit and smoke tests

## Setup

```bash
conda create -n nlb python=3.10
conda activate nlb
make setup
```

## Data path contract

The runner resolves data from either:
1. `data_path` in the config
2. `NLB_DATA_DIR` plus a dataset-specific default subpath

For `mc_maze`, the expected default layout is:

```bash
$NLB_DATA_DIR/000128/sub-Jenkins
```

The pipeline validates that matching NWB files exist for:

```bash
<resolved_data_path>/<data_prefix>*.nwb
```

## Downloading data

Recommended:

```bash
python -m scripts.get_data --dataset mc_maze --out data/raw
export NLB_DATA_DIR=$(pwd)/data/raw
```

## Running

```bash
make run
```

Equivalent:

```bash
python -m scripts.run_experiment --config configs/mc_maze_smoothing.yaml
```

## Switching model families

Set `model_type` in `configs/mc_maze_smoothing.yaml` to one of:

- `smoothing`
- `pca_latent_regression`
- `ridge_direct`

Model-specific config keys:

- `smoothing`
  - reference: `baseline.kern_sd_ms`, `baseline.alpha`
  - CV grid: `improvement.kern_sd_grid`, `improvement.alpha_grid`
- `pca_latent_regression`
  - reference: `baseline.n_components`, `baseline.ridge_alpha`
  - CV grid: `improvement.n_components_grid`, `improvement.ridge_alpha_grid`
- `ridge_direct`
  - reference: `baseline.ridge_alpha`
  - CV grid: `improvement.ridge_alpha_grid`

## Current checked-in results

The current saved results in `results/mc_maze/` come from `pca_latent_regression` with:

- reference params: `n_components=10`, `ridge_alpha=0.1`
- selected params: `n_components=10`, `ridge_alpha=0.1`
- co-bps: `0.003868`
- vel R2: `0.075521`
- psth R2: `-24.147990`

These results are stored in:

- `results/mc_maze/metrics.csv`
- `results/mc_maze/ablation.csv`
- `results/mc_maze/summary.md`

The prediction `.h5` files and full run metadata are intentionally ignored and are not committed.

## Tests

```bash
make test
```

Coverage includes:

- import smoke tests
- data path resolution tests
- synthetic evaluation smoke tests
- shape and parameter-sensitivity tests for all implemented model families

## Notes

- The output filenames remain `baseline_predictions.h5` and `improved_predictions.h5` for compatibility, but logs and summaries describe runs as `reference` and `selected`.
- This repository is intended as a compact, laptop-feasible NLB project rather than a full benchmark sweep.
