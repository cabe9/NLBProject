# Pipeline architecture

A single experiment run is a straight pipeline: load NWB, build tensors, fit a model, convert predictions to held-out neuron rates, score with the official NLB evaluator, and write tracked artifacts.

```
┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────────────┐   ┌───────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│   NWB    │──▶│  NWBDataset│──▶│ tensors │──▶│ temporal features│──▶│ model │──▶│  output head  │──▶│ evaluation │──▶│ reporting  │
│ (.nwb)   │   │ (resample) │   │ (np)    │   │  (lag / scale)   │   │ (fit) │   │  (log_link…)  │   │ (nlb_tools)│   │ (csv/md/svg)│
└──────────┘   └────────────┘   └─────────┘   └──────────────────┘   └───────┘   └───────────────┘   └────────────┘   └────────────┘
```

## Stage-by-stage

1. **NWB file** — raw benchmark data, one file per session. Resolved from `data_path` in the config, or from `NLB_DATA_DIR` plus a dataset-specific default subpath. See `src/nlb_project/data_paths.py`.
2. **`NWBDataset`** — the official `nlb_tools.nwb_interface.NWBDataset` loader. Handles resampling to the 5 ms NLB bin width and stores spike counts on the dataset.
3. **Tensors** — `nlb_tools.make_tensors.make_train_input_tensors` / `make_eval_input_tensors` produce the train/held-in and eval/held-out spike-count arrays in the exact shape NLB expects.
4. **Temporal features** — optional lagged-history construction plus train-only input transforms (`sqrt`, `sqrt_zscore`, etc.) in `src/nlb_project/models/temporal_features.py`. The lag buffer and z-score statistics are fit on train only.
5. **Model** — one model family from `src/nlb_project/models/`, dispatched via the `MODEL_REGISTRY` in `src/nlb_project/model_registry.py`. Each family exports a `fit_predict_*` function that returns held-in + held-out predictions.
6. **Output head** — a shared rate readout in `src/nlb_project/models/output_head.py`. Default is `log_link` (ridge on `log(count + log_offset)`, exponentiated at inference); `linear` and `poisson_glm` are kept for ablations. Every model reuses the same head so comparisons are apples-to-apples.
7. **Evaluation** — `nlb_tools.evaluation.evaluate` computes `co-bps`, `vel R2`, and `psth R2` (when available) against the official splits. The repo never reimplements metrics.
8. **Reporting** — per-run `metrics.csv`, `run_metadata.json`, and optional portfolio artifacts (`model_comparison.csv`/`.md`/`.svg`, `model_diagnostics.svg`, `experiment_log.md`). See `src/nlb_project/reporting.py`.

## Control flow

- `scripts/run_experiment.py` is the single CLI entrypoint for a run.
- It loads a YAML config through `nlb_project.config.load_config`, which is strictly validated against the model registry.
- `nlb_project.pipeline.run_experiment` dispatches the model via `MODEL_REGISTRY[cfg.model]`, builds the parameter grid from `ModelSpec.sweep_axes`, runs cross-validation, and writes artifacts.
- `scripts/generate_portfolio_artifacts.py` never runs models — it re-reads the tracked `metrics.csv` files and rebuilds comparison CSV / Markdown / SVG deterministically.

## What's intentionally NOT in the pipeline

- No benchmark-split logic (handed off entirely to `nlb_tools`).
- No custom metric implementations.
- No global data-mutating state; each run writes to its own output directory.
- No network calls at evaluation time; data is fetched once up front via `scripts/get_data.py`.
