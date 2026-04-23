# Neural Latents Benchmark (`mc_maze`) — lagged PCA beats static baselines

[![CI](https://github.com/cabe9/NLBProject/actions/workflows/ci.yml/badge.svg)](https://github.com/cabe9/NLBProject/actions/workflows/ci.yml)

Short neural history is the single biggest driver of `co-bps` on the NLB'21 `mc_maze` dataset in this comparison. A lagged PCA latent regression reaches `co-bps = 0.027` with `vel R² = 0.36`, from a static baseline near zero — all under the official `nlb_tools` evaluation path.

## Results at a glance

Scored under the `log_link` rate readout on the `mc_maze` train/val split. Full table in [`results/benchmark_runs/model_comparison.md`](results/benchmark_runs/model_comparison.md); source `metrics.csv` files under [`results/benchmark_runs/`](results/benchmark_runs/).

| Model | Role | co-bps | vel R² |
|---|---|---:|---:|
| static PCA latent regression | reference | −0.0068 | 0.0756 |
| static direct ridge | reference | 0.0017 | 0.0769 |
| lagged direct ridge (5 bins) | reference | 0.0215 | 0.1986 |
| lagged reduced-rank regression (selected) | selected | 0.0283 | 0.2316 |
| lagged PCA latent regression (5 bins) | reference | 0.0166 | 0.2400 |
| **lagged PCA latent regression (selected history)** | **headline** | **0.0266** | **0.3648** |

![co-bps comparison](results/benchmark_runs/model_comparison.svg)

**Takeaways**

- Every lagged model beats every static model on `co-bps`. Temporal context dominates static latent dimensionality.
- RRR and lagged PCA tie on `co-bps` within `~0.002`. The discriminator is `vel R²`, where lagged PCA's unsupervised bottleneck captures more behaviour-aligned structure.
- An earlier version of this repo reported `lagged_ridge_direct ≈ −0.43 co-bps`; that number was a clipped-Gaussian readout artefact, not a model failure. Full writeup: [`docs/output_head_postmortem.md`](docs/output_head_postmortem.md).

## Reproduce in 5 commands

```bash
conda create -n nlb python=3.10 && conda activate nlb
make setup
python -m scripts.get_data --dataset mc_maze --out data/raw && export NLB_DATA_DIR=$(pwd)/data/raw
make run                 # runs the validated lagged PCA experiment
make portfolio-artifacts # rebuilds comparison CSV / Markdown / SVG from saved metrics
```

Tests: `make test`. Lint/format: `make lint` / `make format`.

## Next clickable thing

Start here, not in a source file:

- **[`notebooks/results_walkthrough.ipynb`](notebooks/results_walkthrough.ipynb)** — rendered walkthrough of the comparison table and plots (90-second skim).
- **[`docs/architecture.md`](docs/architecture.md)** — pipeline diagram and stage-by-stage control flow.
- **[`docs/models.md`](docs/models.md)** — one-paragraph description of each model family.
- **[`docs/output_head_postmortem.md`](docs/output_head_postmortem.md)** — how a clipped rate readout silently rewrote the scientific narrative, and how it was fixed.

## Repo layout

```
configs/               experiment YAML configs (benchmark suite + the selected config)
docs/                  architecture notes, model descriptions, output-head postmortem
notebooks/             results_walkthrough.ipynb — rendered comparison
results/               tracked per-run metrics.csv + portfolio comparison artifacts
scripts/               run_experiment.py, get_data.py, generate_portfolio_artifacts.py
src/nlb_project/
  config.py            typed, fail-fast config loading
  model_registry.py    declarative ModelSpec entries; single source of truth for sweeps
  pipeline.py          orchestration: load → tensors → fit → head → evaluate → artifacts
  models/              fit_predict_* implementations (one file per family) + output_head
  reporting.py         deterministic rebuild of portfolio comparison artifacts
tests/                 pytest suite (shape, CV, config, data contract, smoke)
```

## Benchmark discipline

What this repo does **not** change:

- benchmark splits
- metric definitions
- evaluation code (handed off to `nlb_tools.evaluation.evaluate`)
- data-loading conventions (uses `nlb_tools.nwb_interface.NWBDataset` + `nlb_tools.make_tensors`)

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline.

## Suggested GitHub topics

When you publish this repo, add: `neuroscience`, `machine-learning`, `neural-decoding`, `time-series`, `reproducibility`, `benchmarking`.

## Citation

```bibtex
@software{telander_nlb_project,
  author  = {Caleb Telander},
  title   = {Neural Latents Benchmark (mc_maze) Analysis},
  year    = {2026},
  url     = {https://github.com/cabe9/NLBProject}
}
```

The underlying benchmark and evaluation code come from [`nlb_tools`](https://github.com/neurallatents/nlb_tools); please also cite the Neural Latents Benchmark if you use these results.

## License

[MIT](LICENSE).
