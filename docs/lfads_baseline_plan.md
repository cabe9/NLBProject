# LFADS / AutoLFADS baseline plan (MC_Maze)

## Purpose

Add **LFADS** (`lfads-torch`) as a **separate model-family baseline** beside the
existing STNDT-lite stack. This is **not** another STNDT-lite screen.

- Current STNDT-lite local public-test headline: **0.3830 co-bps** (5 ms, mixed ensemble).
- LFADS upstream MC_Maze example: **20 ms** (`mc_maze-20ms-val.h5`).
- **Never compare 5 ms and 20 ms numbers without explicit bin-size labels.**
- **No public-test** until train/val conversion and evaluation are validated and you
  explicitly approve.

## Milestones

### Done (smoke + evaluation bridge)

1. Isolated env plan (`lfads-nlb` / `.venv-lfads-nlb`).
2. `external/lfads-torch` clone (gitignored).
3. HDF5 prep + smoke train (checkpoint saved).
4. **Rate export + NLB validation evaluator** (`scripts/export_lfads_rates.py`,
   `scripts/evaluate_lfads_outputs.py`, `scripts/lfads_nlb_bridge.py`).

### Next (not started here)

1. One conservative **20 ms** train/val baseline (single seed, limited epochs, no sweep).
2. Decide whether **5 ms** NWB path is worth the extra format work for STNDT-lite comparability.
3. Public-test only after full val gate + explicit approval.

## Repository layout

| Path | Role |
|------|------|
| `external/lfads-torch/` | Upstream clone (`LFADS_TORCH_DIR`); gitignored |
| `data/lfads/` | Prepared HDF5 + manifests (gitignored under `data/`) |
| `configs/lfads/` | Project notes and dimension tables |
| `scripts/setup_lfads_torch.*` | Clone + install instructions |
| `scripts/prepare_lfads_mc_maze.py` | Reference copy (20 ms) or NWB build (5/20 ms) |
| `scripts/run_lfads_mc_maze_smoke.py` | Tiny training smoke test |
| `scripts/export_lfads_rates.py` | Posterior sampling from checkpoint (no training) |
| `scripts/lfads_nlb_bridge.py` | LFADS HDF5 → NLB user/target dict conversion |
| `scripts/evaluate_lfads_outputs.py` | Alignment checks + `nlb_tools.evaluate` |
| `results/lfads_smoke/` | Smoke run logs, checkpoints, `lfads_outputs/` |

## Environment setup

### Preferred: conda `lfads-nlb` (Python 3.9)

```powershell
cd C:\Users\david\NLBProject
.\scripts\setup_lfads_torch.ps1
conda activate lfads-nlb
pip install --upgrade pip
pip install -e "$env:LFADS_TORCH_DIR\external\lfads-torch" --no-deps
# If LFADS_TORCH_DIR unset, use external\lfads-torch under repo root
pip install -e "external\lfads-torch" --no-deps
pip install torch==1.13.1 pytorch-lightning==1.6.0 torchmetrics==0.7.2 hydra-core==1.3.0 h5py "numpy<2" scikit-learn matplotlib
pip install "pandas==1.3.4" nlb-tools==0.0.4
# ray[tune] optional for smoke (single-session stub); required later for AutoLFADS PBT
python -c "import torch; import lfads_torch; import nlb_tools; print('ok')"
```

**Notes:**

- Install `lfads-torch` with `--no-deps` (upstream pins `ray[tune]==2.1.0`, often unavailable on pip).
- Pin `pandas==1.3.4` for `nlb-tools==0.0.4` compatibility.
- Single-session **smoke** stubs `ray.tune` at import time; install real `ray[tune]` only for PBT.

### Without conda (Windows fallback)

```powershell
py -3.10 -m venv .venv-lfads-nlb
.\.venv-lfads-nlb\Scripts\Activate.ps1
# same pip lines as above
```

Do **not** install LFADS into the main STNDT-lite `nlb` environment (`requires-python 3.10`,
different torch stack).

## Data preparation

### A. 20 ms reference (easiest — matches lfads-torch `nlb_mc_maze`)

Copies bundled `datasets/mc_maze-20ms-val.h5` after clone:

```powershell
conda activate lfads-nlb   # or your venv
python scripts/prepare_lfads_mc_maze.py --source reference --write-smoke-subset
```

Outputs:

- `data/lfads/mc_maze_20ms_val.h5`
- `data/lfads/mc_maze_20ms_val_manifest.json`
- `data/lfads/mc_maze_20ms_val_smoke.h5` (8 train + 4 val trials)

Reference shapes:

| Key | Shape |
|-----|-------|
| `train_encod_data` | (1721, 35, 137) |
| `train_recon_data` | (1721, 45, 182) |
| `valid_encod_data` | (574, 35, 137) |
| `valid_recon_data` | (574, 45, 182) |

Config mapping (upstream `configs/model/nlb_mc_maze.yaml`):

- `encod_data_dim`: 137
- `encod_seq_len`: 35
- `recon_seq_len`: 45
- `readout` `out_features`: 182
- `datamodule.batch_size`: 256 upstream; **use 32–64 for 5 ms on RTX 3080** when that track exists

### B. 5 ms from local NWB (STNDT-lite-aligned bin width)

Requires `NLB_DATA_DIR` or `--data-path` (same layout as main pipeline):

```powershell
$env:NLB_DATA_DIR = "C:\Users\david\NLBProject\data\raw"
python scripts/prepare_lfads_mc_maze.py --bin-size-ms 5 --source nwb
```

Update `configs/lfads/mc_maze_5ms_from_nwb.yaml` and lfads model YAML dimensions from the
written `*_manifest.json` before any real training.

## Smoke test (training)

```powershell
python scripts/run_lfads_mc_maze_smoke.py --max-epochs 1 --batch-size 4
```

Pass criteria:

- Imports: `torch`, `lfads_torch`, `nlb_tools`
- Data loads; no NaNs in encod/recon keys
- Model trains ≥1 epoch; checkpoint under `results/lfads_smoke/<run_id>/lfads_run/lightning_checkpoints/`
- `manifest.json` records shapes

Training alone does **not** write `lfads_output*.h5` (posterior sampling is off in smoke).

## Rate export (posterior sampling)

Uses `lfads_torch.post_run.analysis.run_posterior_sampling` (same path as upstream
`run_model(..., do_posterior_sample=True)`): loads checkpoint, averages `num_samples`
forward passes, writes `train_output_params` / `valid_output_params` into a copy of the
data HDF5.

```powershell
python scripts/export_lfads_rates.py `
  --run-dir results/lfads_smoke/20260603T074245Z `
  --num-samples 2
```

Outputs:

- `results/lfads_smoke/<run_id>/lfads_outputs/lfads_output_mc_maze_20ms_val_smoke.h5`
- `lfads_outputs/export_manifest.json`

Expected rate shapes (20 ms MC_Maze):

| Key | Shape (smoke subset) | Shape (full val file) |
|-----|----------------------|------------------------|
| `train_output_params` | (8, 45, 182) | (1721, 45, 182) |
| `valid_output_params` | (4, 45, 182) | (574, 45, 182) |

Split mapping (`scripts/lfads_nlb_bridge.py`, mirrors AutoLFADS `post_lfads_prep.py`):

- Channels `0:137` = held-in, `137:182` = held-out
- Time `0:35` = observed (encod length); `35:45` = forward prediction
- `valid_output_params` → NLB `eval_rates_*` (val trials)
- `train_output_params` → NLB `train_rates_*` (train trials)
- Targets for co-bps: `valid_recon_data[:, :35, 137:]` held-out spikes

## NLB validation evaluation

```powershell
python scripts/evaluate_lfads_outputs.py `
  --run-dir results/lfads_smoke/20260603T074245Z
```

Writes `lfads_outputs/lfads_output_*_nlb_eval.json` with:

- Shape alignment checks (user rates vs targets)
- Metrics from `nlb_tools.evaluation.evaluate` when aligned

Dataset keys for **20 ms**: `mc_maze_20` → metrics under `mc_maze_20_split`.

### Smoke evaluation status (2026-06-03)

**Works end-to-end** on `20260603T074245Z`:

- Export + evaluate complete; finite rates; alignment `ok: true`
- Example smoke metrics (1 epoch, 8+4 trials): `co-bps ≈ -16.38`, `vel R2 ≈ -0.72`
- **Not meaningful performance** — only plumbing verification
- **psth R2 omitted** on trial subsets when `valid_cond_idx` references full-dataset trial IDs

### Before a real 20 ms train/val baseline

- [ ] Train on **full** `data/lfads/mc_maze_20ms_val.h5` (not smoke subset)
- [ ] Export with more posterior samples (e.g. 20–50)
- [ ] Re-evaluate; expect finite `co-bps` on val (quality TBD, not vs 0.3830)
- [ ] Optional: rebuild targets from NWB via `make_eval_target_tensors` for exact NLB parity
- [ ] 5 ms path only if fair STNDT-lite comparison is required

## Recommended path: 20 ms first, then decide on 5 ms

**20 ms first** — fastest working LFADS baseline (model-family demo, not comparable to 5 ms headline).

**5 ms from NWB** — better for fair comparison to STNDT-lite `0.3830`, more preprocessing/debug risk.

## Policy

- LFADS work does not change STNDT-lite training code paths.
- Do not overwrite `results/benchmark_runs/*`.
- Do not run PBT / long training / public-test from this milestone.
- Full AutoLFADS (PBT) is a later step after single-session smoke is stable.

## Related docs

- `configs/lfads/README.md` — dimension cheat sheet
- `AGENTS.md` — short guardrail pointer (local)
