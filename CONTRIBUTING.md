# Contributing

This project is a reproducible benchmark workflow. Contributions should improve clarity, reliability, or model quality without weakening comparability.

## Setup

Python support is currently **3.10 only**.

```bash
conda create -n nlb python=3.10 -y
conda activate nlb
make setup
```

If you want to run full benchmark configs, download data once:

```bash
python -m scripts.get_data --dataset mc_maze --out data/raw
export NLB_DATA_DIR="$(pwd)/data/raw"
```

## Development commands

- Lint: `make lint`
- Format: `make format`
- Tests: `make test`
- Type check: `make typecheck`
- Rebuild portfolio artifacts from tracked metrics: `make portfolio-artifacts`

Recommended pre-PR check:

```bash
make lint && ruff format --check . && make typecheck && make test
```

## Branch and PR expectations

- Use a short, descriptive branch name (`docs/onboarding`, `test/pipeline-integration`, etc.).
- Keep PRs focused; avoid mixing scientific changes, tooling changes, and docs rewrites in one PR.
- In your PR description, include:
  - what changed and why,
  - commands you ran locally,
  - whether tracked artifacts changed.
- If benchmark metrics change, explain exactly why and which config/run caused it.

## Adding a new model family

1. Implement a `fit_predict_*` function in `src/nlb_project/models/`.
2. Export it in `src/nlb_project/models/__init__.py`.
3. Register it in `src/nlb_project/model_registry.py` with:
   - `baseline_params`,
   - `sweep_axes`,
   - any `improvement_overrides`,
   - `uses_rate_head` / `passes_log_offset` flags as needed.
4. Add or update config(s) in `configs/`.
5. Add tests (shape/contract coverage at minimum, plus behavior checks when feasible).
6. Run the full validation commands before opening a PR.

The pipeline should stay model-agnostic: avoid adding per-model branches in `src/nlb_project/pipeline.py` when a registry declaration can express the behavior.

## Files and outputs that should not be edited casually

- `results/benchmark_runs/**` (tracked portfolio artifacts)
- `results/mc_maze/**` (tracked main run outputs)
- `configs/*.yaml` used for benchmark claims
- docs or README sections that state headline metrics

Do not hand-edit generated CSV/Markdown/SVG artifacts. Regenerate via scripts/Make targets so provenance stays clear.

## Reproducibility expectations

- Keep experiment behavior deterministic where possible (seeded paths are preferred).
- Do not change benchmark split logic or metric definitions; those come from `nlb_tools`.
- Do not introduce hidden defaults that alter model parameters silently.
- Any scientific claim should map to:
  - a committed config in `configs/`,
  - a committed artifact under `results/`,
  - and a reproducible command path.
