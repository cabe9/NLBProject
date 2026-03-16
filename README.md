# Neural Latents Benchmark (NLB'21) Project

This repo is a reproducible NLB'21 project focused on `mc_maze`.

It currently supports three model families:
1. `smoothing`: Gaussian smoothing of held-in spikes plus Poisson readout for held-out neurons.
2. `pca_latent_regression`: PCA on held-in activity, then Ridge regression from latent state to held-out rates.
3. `ridge_direct`: direct multi-target Ridge regression from held-in rates to held-out rates.

Primary metric: `co-bps`. Secondary metrics: `vel R2` and `psth R2` when available.

## Project structure

- `src/nlb_project/`: config loading, data path validation, models, and pipeline code
- `scripts/run_experiment.py`: main experiment entrypoint
- `scripts/get_data.py`: dataset download helper
- `configs/mc_maze_smoothing.yaml`: current experiment config
- `results/`: saved metrics, summaries, metadata, and prediction artifacts
- `tests/`: unit and smoke tests

## Setup

```bash
conda create -n nlb python=3.10
conda activate nlb
make setup
```

This installs the project into the conda env and keeps heavy dependencies out of `base`.

## Data path contract

The runner expects NWB files under either:
1. `data_path` in the config, or
2. `NLB_DATA_DIR` plus a dataset-specific subpath

For `mc_maze`, the default expected location is:

```bash
$NLB_DATA_DIR/000128/sub-Jenkins
```

The pipeline validates that matching files exist for:

```bash
<resolved_data_path>/<data_prefix>*.nwb
```

If the path cannot be resolved, or if no NWB files match, the run fails early with a clear error.

## Getting data

Recommended:

```bash
python -m scripts.get_data --dataset mc_maze --out data/raw
export NLB_DATA_DIR=$(pwd)/data/raw
```

Or download the dataset yourself and point `NLB_DATA_DIR` at the raw root.

## Running experiments

Current default config:

```yaml
model_type: pca_latent_regression
dataset_name: mc_maze
```

Run with:

```bash
make run
```

or:

```bash
python -m scripts.run_experiment --config configs/mc_maze_smoothing.yaml
```

The pipeline:
1. loads NWB data with `nlb_tools.nwb_interface.NWBDataset`
2. builds train/eval/target tensors with official `nlb_tools.make_tensors` helpers
3. runs a reference parameter set
4. runs CV to choose a selected parameter set for the active model type
5. writes prediction `.h5` files and local evaluation metrics

## Switching models

Set `model_type` in `configs/mc_maze_smoothing.yaml` to one of:

- `smoothing`
- `pca_latent_regression`
- `ridge_direct`

Model-specific params:

- `smoothing`
  - reference: `baseline.kern_sd_ms`, `baseline.alpha`
  - CV grid: `improvement.kern_sd_grid`, `improvement.alpha_grid`
- `pca_latent_regression`
  - reference: `baseline.n_components`, `baseline.ridge_alpha`
  - CV grid: `improvement.n_components_grid`, `improvement.ridge_alpha_grid`
- `ridge_direct`
  - reference: `baseline.ridge_alpha`
  - CV grid: `improvement.ridge_alpha_grid`

## Outputs

Runs write to `results/mc_maze/`:

- `metrics.csv`
- `ablation.csv`
- `summary.md`
- `run_metadata.json`
- `predictions/baseline_predictions.h5`
- `predictions/improved_predictions.h5`

The filenames stay fixed for compatibility, even though logs refer to `reference` vs `selected` params.

## Evaluation

The project uses official `nlb_tools` evaluation utilities:

- `make_train_input_tensors`
- `make_eval_input_tensors`
- `make_eval_target_tensors`
- `evaluate`

For `mc_maze`, the local evaluation split key is `mc_maze_split`.

## Tests

Run:

```bash
make test
```

The test suite covers:

- import smoke tests
- data path resolution
- synthetic evaluation smoke tests
- model output shape checks
- parameter-sensitivity checks for smoothing, PCA latent regression, and ridge direct

## Current status

The codebase is beyond the original smoothing-only scaffold. The active config and latest saved results currently point at `pca_latent_regression`, while the other two model families are implemented and test-covered.
