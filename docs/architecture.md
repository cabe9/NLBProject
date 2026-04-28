# Pipeline architecture

A single experiment run is a straight pipeline: load NWB, build tensors, fit a model, convert predictions to held-out neuron rates, score with the official NLB evaluator, and write tracked artifacts.

```
┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────────────┐   ┌───────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│   NWB    │──▶│  NWBDataset│──▶│ tensors │──▶│ temporal features│──▶│ model │──▶│  output head  │──▶│ evaluation │──▶│ reporting  │
│ (.nwb)   │   │ (resample) │   │ (np)    │   │  (lag / scale)   │   │ (fit) │   │  (log_link…)  │   │ (nlb_tools)│   │ (csv/md/svg)│
└──────────┘   └────────────┘   └─────────┘   └──────────────────┘   └───────┘   └───────────────┘   └────────────┘   └────────────┘
```

## Stage-by-stage

1. **NWB file** — raw benchmark data, one file per session. Resolved from `data_path` in the config, or from `NLB_DATA_DIR` plus a dataset-specific default subpath. See `src/nlb_project/data_contract.py` (`resolve_data_path`).
2. **`NWBDataset`** — the official `nlb_tools.nwb_interface.NWBDataset` loader. Handles resampling to the 5 ms NLB bin width and stores spike counts on the dataset.
3. **Tensors** — `nlb_tools.make_tensors.make_train_input_tensors` / `make_eval_input_tensors` produce the train/held-in and eval/held-out spike-count arrays in the exact shape NLB expects.
4. **Temporal features** — optional lagged-history construction plus train-only input transforms (`sqrt`, `sqrt_zscore`, etc.) in `src/nlb_project/models/temporal_features.py`. The lag buffer and z-score statistics are fit on train only.
5. **Model** — one model family from `src/nlb_project/models/`, dispatched via the `MODEL_REGISTRY` in `src/nlb_project/model_registry.py`. Each family exports a `fit_predict_*` function that returns held-in + held-out predictions.
6. **Output head** — a shared rate readout in `src/nlb_project/models/output_head.py`. Default is `log_link` (ridge on `log(count + log_offset)`, exponentiated at inference); `linear` and `poisson_glm` are kept for ablations. Every model reuses the same head so comparisons are apples-to-apples.
7. **Evaluation** — `nlb_tools.evaluation.evaluate` computes `co-bps`, `vel R2`, and `psth R2` (when available) against the official splits. The repo never reimplements metrics.
8. **Reporting** — per-run `metrics.csv`, `run_metadata.json`, and optional portfolio artifacts (`model_comparison.csv`/`.md`/`.svg`, `model_diagnostics.svg`, `experiment_log.md`). `run_metadata.json` records the config path, git state, Python/package versions, resolved data path, and prediction artifact hashes. See `src/nlb_project/reporting.py` and `src/nlb_project/run_metadata.py`.

## Control flow

- `scripts/run_experiment.py` is the single CLI entrypoint for a run.
- It loads a YAML config through `nlb_project.config.load_config`, which is strictly validated against the model registry.
- `nlb_project.pipeline.run_full_experiment` dispatches the model via the selected `ModelSpec`, builds the parameter grid from `ModelSpec.sweep_axes`, runs cross-validation, and writes artifacts.
- `scripts/generate_portfolio_artifacts.py` never runs models — it re-reads the tracked `metrics.csv` files and rebuilds comparison CSV / Markdown / SVG deterministically.
- `scripts/verify_results.py` audits result artifacts without rerunning models. It always checks tracked `metrics.csv` and `model_comparison.csv` files, and verifies local `run_metadata.json` / prediction hashes when those full-run artifacts are present.

## Data/config/run/artifact flow

1. **Config authoring** (`configs/*.yaml`)  
   A config selects `model_type`, baseline params, and CV sweep grids.
2. **Config validation** (`src/nlb_project/config.py`)  
   `load_config()` enforces required keys and rejects unknown keys against `MODEL_REGISTRY`.
3. **Run execution** (`scripts/run_experiment.py` -> `src/nlb_project/pipeline.py`)  
   The run resolves data location, loads/resamples NWB data, performs CV model selection, scores with `nlb_tools.evaluation.evaluate`, and saves predictions/metrics.
4. **Run artifacts** (`results/mc_maze/` by default config)  
   The run writes `metrics.csv`, `ablation.csv`, `summary.md`, `run_metadata.json`, and prediction HDF5 files.
5. **Portfolio artifacts** (`scripts/generate_portfolio_artifacts.py`)
   A separate reporting pass re-reads committed `metrics.csv` files to regenerate `results/benchmark_runs/model_comparison.csv`, `.md`, `.svg`, plus diagnostics SVG and experiment log.
6. **Provenance verification** (`scripts/verify_results.py`)
   `make verify-results` fails when tracked metrics are malformed, when local metadata disagrees with metrics, or when the generated comparison CSV is stale relative to the reporting manifest. Missing local metadata or prediction files are warnings because those full-run artifacts are ignored by git.

## What's intentionally NOT in the pipeline

- No benchmark-split logic (handed off entirely to `nlb_tools`).
- No custom metric implementations.
- No global data-mutating state; each run writes to its own output directory.
- No network calls at evaluation time; data is fetched once up front via `scripts/get_data.py`.
