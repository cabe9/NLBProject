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

1. Isolated env plan (`lfads-nlb` / `.venv-lfads-nlb`; local env: `.venv-lfads-smoke-test` with `torch 1.13.1+cu117`).
2. `external/lfads-torch` clone (gitignored).
3. HDF5 prep + smoke train (checkpoint saved).
4. **Rate export + NLB validation evaluator** (`scripts/export_lfads_rates.py`,
   `scripts/evaluate_lfads_outputs.py`, `scripts/lfads_nlb_bridge.py`).
5. **Single-seed 20 ms train/val baselines** (50 and 100 epochs, full HDF5) — see
   [Completed 20 ms validation baselines](#completed-20-ms-validation-baselines-2026-06-03).
   Current best: **0.3606 co-bps** @ 100 epochs (`20260603T201838Z`).

### Next (controlled runs only)

1. Optional **batch-size 64** probe if GPU memory allows (one OOM retry to 32; not a sweep).
2. Optional **second seed** at 50–100 epochs if reproducibility matters.
3. **5 ms** pipeline validated through probe + salvage; long-train unstable — see
   [5 ms LFADS pipeline](#5-ms-lfads-pipeline-2026-06-04). Further 5 ms work needs explicit
   stability knobs, not another unattended 50-epoch job.
4. **No 200-epoch** scaling without a new structural reason (diminishing returns vs 50→100).
5. **No public-test** until explicit approval; never compare 20 ms scores to the 5 ms headline without bin-size labels.

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

Requires `NLB_DATA_DIR` or `--data-path` (same layout as main pipeline). Use
`--skip-psth` — full PSTH at 5 ms OOMs (~10 GB); evaluation omits **psth R2** when PSTH
is absent.

```powershell
$env:NLB_DATA_DIR = "C:\Users\david\NLBProject\data\raw"
python scripts/prepare_lfads_mc_maze.py --bin-size-ms 5 --source nwb --skip-psth --write-smoke-subset
```

Outputs: `data/lfads/mc_maze_5ms_val.h5`, `mc_maze_5ms_val_smoke.h5`, manifest. Shapes
(2026-06-04 build):

| Key | Shape |
|-----|-------|
| `train_encod_data` | (1721, 140, 137) |
| `train_recon_data` | (1721, 180, 182) |
| `valid_encod_data` | (574, 140, 137) |
| `valid_recon_data` | (574, 180, 182) |

Model dims: `encod_data_dim` 137, `encod_seq_len` 140, `recon_seq_len` 180,
`readout_out_features` 182 (`configs/lfads/mc_maze_5ms_from_nwb.yaml`).

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

### First 20 ms train/val baseline (done 2026-06-03)

- [x] Train on **full** `data/lfads/mc_maze_20ms_val.h5` (not smoke subset)
- [x] Export with posterior samples (20 on baseline run)
- [x] Re-evaluate; finite `co-bps` on val — see baseline table below
- [ ] Optional: rebuild targets from NWB via `make_eval_target_tensors` for exact NLB parity
- [x] 5 ms path: probe + salvage documented — see [5 ms LFADS pipeline](#5-ms-lfads-pipeline-2026-06-04)

## Completed 20 ms validation baselines (2026-06-03)

All metrics are **20 ms MC_Maze train/val** via `nlb_tools.evaluate` on
`data/lfads/mc_maze_20ms_val.h5` targets (`mc_maze_20_split`). **Not public-test.**
**Not comparable** to the STNDT-lite **5 ms** local public-test headline (**0.3830 co-bps**).

Env: `.venv-lfads-smoke-test`, CUDA RTX 3080, `scripts/run_lfads_mc_maze_smoke.py` on full HDF5.

### 2-epoch probe (plumbing + under-trained check)

| Field | Value |
|-------|-------|
| Run dir | `results/lfads_smoke/20260603T092356Z/` |
| Train | 2 epochs, batch 16, full HDF5 |
| Export / eval | OK (5 posterior samples); alignment OK |

| Metric | Value |
|--------|------:|
| co-bps | 0.1978 |
| fp-bps | 0.1188 |
| vel R2 | 0.7931 |
| psth R2 | 0.2721 |

Eval JSON: `lfads_outputs/lfads_output_mc_maze_20ms_val_nlb_eval.json`

### 50-epoch baseline (first real single-seed baseline)

| Field | Value |
|-------|-------|
| Run dir | `results/lfads_smoke/20260603T094126Z/` |
| Data | `data/lfads/mc_maze_20ms_val.h5` (1721 train / 574 val trials) |
| Train | 50 epochs, **batch size 32**, ~**15.7 min** wall time, CUDA |
| Export | **20** posterior samples, ~**19.5 min**; finite `train/valid_output_params` |
| Issues | No NaNs, no manifest warnings; **PSTH included** in evaluation |

| Metric | Value |
|--------|------:|
| co-bps | **0.3499** |
| fp-bps | 0.2402 |
| vel R2 | 0.8967 |
| psth R2 | 0.5843 |

Artifacts:

- Checkpoint: `lfads_run/lightning_checkpoints/49-2650.ckpt`
- Rates: `lfads_outputs/lfads_output_mc_maze_20ms_val.h5`
- Eval JSON: `lfads_outputs/lfads_output_mc_maze_20ms_val_nlb_eval.json`

### 100-epoch baseline (current best single-seed, 20 ms)

| Field | Value |
|-------|-------|
| Run dir | `results/lfads_smoke/20260603T201838Z/` |
| Data | `data/lfads/mc_maze_20ms_val.h5` (1721 train / 574 val trials) |
| Train | 100 epochs, **batch size 32**, ~**31.7 min** wall time, CUDA |
| Export | **20** posterior samples, ~**20.0 min**; finite `train/valid_output_params` |
| Issues | No NaNs, no manifest warnings; **PSTH included** in evaluation |

| Metric | Value |
|--------|------:|
| co-bps | **0.3606** |
| fp-bps | 0.2525 |
| vel R2 | 0.8946 |
| psth R2 | 0.5818 |

Artifacts:

- Checkpoint: `lfads_run/lightning_checkpoints/98-5247.ckpt` (export used this best ckpt)
- Rates: `lfads_outputs/lfads_output_mc_maze_20ms_val.h5`
- Eval JSON: `lfads_outputs/lfads_output_mc_maze_20ms_val_nlb_eval.json`

**vs 50 epochs (`20260603T094126Z`):**

| Metric | Δ (100 − 50) |
|--------|-------------|
| co-bps | **+0.0107** (0.3499 → 0.3606) |
| fp-bps | +0.0123 |
| vel R2 | −0.0021 (flat / slightly lower) |
| psth R2 | −0.0025 (flat / slightly lower) |

### Interpretation

- This is a **valid first LFADS 20 ms validation baseline** on the bundled lfads-torch HDF5 split.
- The LFADS track is **more than smoke plumbing**: export → NLB evaluate works on full data with PSTH.
- **0.3606 co-bps (20 ms, 100 epochs)** is the **current best single-seed** result; **0.3499 (50 epochs)** is the prior reference.
- Neither must be read against **0.3830 (5 ms STNDT-lite)** without matching bin size and data contract.
- Probe (2 epoch) → 50 epoch gained ~**+0.15 co-bps**; 50 → 100 gained only **+0.0107 co-bps** while roughly doubling train time — **epoch scaling shows diminishing returns**.
- **Do not run 200 epochs** without a new structural reason.
- Further work should be **controlled single runs** (batch-64 probe, optional second seed, or 5 ms path), not broad sweeps or uncontrolled long loops.

### Recommended next experiments

| Option | Recommendation |
|--------|----------------|
| **200+ epochs** | No — diminishing returns after 100 vs 50 |
| **batch 64 probe** | Next meaningful knob if GPU memory allows; one OOM retry to 32 |
| **Second seed** | Optional reproducibility check at 50–100 epochs |
| **5 ms LFADS** | Pipeline valid; long-train unstable — stability knobs only (see 5 ms section) |
| **8-hour / PBT run** | No |
| **Hyperparameter sweep** | Not yet |
| **Public-test** | No |

## 5 ms LFADS pipeline (2026-06-04)

All metrics below are **5 ms MC_Maze train/val** via `nlb_tools.evaluate` on
`data/lfads/mc_maze_5ms_val.h5` (`mc_maze_split`). **Not public-test.**
**Not comparable** to the STNDT-lite **5 ms** local public-test headline (**0.3830 co-bps**)
without explicit bin-size and model-family labels. **20 ms LFADS** (**0.3606 co-bps** @ 100
epochs) remains the validated LFADS baseline for now.

### Pipeline status

- **5 ms HDF5 creation succeeded** (`prepare_lfads_mc_maze.py --bin-size-ms 5 --skip-psth`).
- **PSTH creation OOMed** at 5 ms (~10 GB); builds use **`--skip-psth`**.
- Shapes (full val HDF5):

| Key | Shape |
|-----|-------|
| `train_encod_data` | (1721, 140, 137) |
| `train_recon_data` | (1721, 180, 182) |
| `valid_encod_data` | (574, 140, 137) |
| `valid_recon_data` | (574, 180, 182) |

### Successful checks (train/val only)

| Step | Run dir | Result |
|------|---------|--------|
| Skip-train / shape check | `20260604T055845Z` | OK |
| 1-epoch smoke train + export/eval | `20260604T055849Z` | OK (plumbing) |
| 2-epoch full-HDF5 probe | `20260604T055919Z` | OK; alignment OK |

**2-epoch probe (5 ms train/val):** **co-bps 0.1343** (under-trained plumbing check, not a
baseline score).

### Failed long-train attempts

| Run dir | Config | Failure |
|---------|--------|---------|
| `20260604T062750Z` | 50 epochs, **batch 8** | ~epoch **18**, IC-posterior numerical blow-up → NaN |
| `20260604T074855Z` | 50 epochs, **batch 4** retry | ~epoch **11**, same failure mode |

Batch-size reduction **did not** fix instability. Checkpoints were not copied into run dirs
on crash; surviving files live under `external/lfads-torch/lightning_checkpoints/`.

### Salvage checkpoint (pre-divergence, batch-8 run)

Export via `scripts/export_lfads_rates.py --checkpoint …` (no new training):

| Field | Value |
|-------|-------|
| Checkpoint | `external/lfads-torch/lightning_checkpoints/14-3225.ckpt` |
| Epoch / step | **14** / **3225** |
| Run dir (manifest / eval) | `results/lfads_smoke/20260604T062750Z/` |
| Rates HDF5 | `lfads_outputs/lfads_output_mc_maze_5ms_val.h5` |
| Outputs | **Finite** (`train/valid_output_params`) |
| Export | 5 posterior samples (~20 min on full HDF5) |

| Metric | Value |
|--------|------:|
| co-bps (val) | **0.2902** |
| fp-bps | 0.1929 |
| vel R2 | 0.8871 |
| psth R2 | *not computed* (`--skip-psth` on 5 ms HDF5) |

**vs 2-epoch probe:** salvage **0.2902** vs **0.1343** co-bps — materially better, but still
**not** a stable full 50-epoch baseline (training diverged before epoch 18; no epoch 15–17
checkpoints on disk).

### Interpretation

- **5 ms LFADS data / export / NLB eval pipeline is valid** end-to-end.
- **Current long-train LFADS recipe is unstable at 5 ms** — do **not** run another unattended
  50-epoch job with this config.
- Further 5 ms LFADS work should use **explicit stability changes**: lower `lr_init`,
  gradient clipping, early stopping / capped epochs, or smaller model — not blind epoch scaling.
- **20 ms LFADS** (**0.3606 co-bps**, 100 epochs) remains the **validated LFADS baseline** until
  5 ms long-train stability is fixed.

## Recommended path

**20 ms** — validated LFADS baseline (**0.3606 co-bps** train/val, 20 ms).

**5 ms** — pipeline proven; fair STNDT-lite bin width, but long-train unstable. Use only with
labeled **5 ms train/val LFADS** scores and stability-first follow-ups.

## Policy

- LFADS work does not change STNDT-lite training code paths.
- Do not overwrite `results/benchmark_runs/*`.
- Do not run PBT / long training / public-test from this milestone.
- Full AutoLFADS (PBT) is a later step after single-session smoke is stable.

## Related docs

- `configs/lfads/README.md` — dimension cheat sheet
- `AGENTS.md` — short guardrail pointer (local)
