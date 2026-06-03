# LFADS / AutoLFADS configs (NLB MC_Maze)

Separate baseline track from STNDT-lite. Do not compare scores across bin sizes
without explicit labeling (`5 ms` vs `20 ms`).

## Files

| File | Purpose |
|------|---------|
| `mc_maze_20ms_smoke.yaml` | Project notes + Hydra override keys for a tiny smoke run |
| `mc_maze_5ms_from_nwb.yaml` | Dimension placeholders when building from NWB at 5 ms |

Upstream reference model/datamodule YAMLs live in `external/lfads-torch/configs/`
after running `scripts/setup_lfads_torch.ps1` (or `.sh`).

## Reference dimensions (lfads-torch `mc_maze-20ms-val.h5`)

| Field | Value |
|-------|-------|
| `bin_size_ms` | 20 |
| `encod_data_dim` | 137 (held-in) |
| `encod_seq_len` | 35 |
| `recon_seq_len` | 45 (35 observed + 10 forward-prediction bins) |
| `recon channels` | 182 (137 held-in + 45 held-out) |
| NLB splits in file | `train` trials → `train_*` keys; `val` trials → `valid_*` keys |
| Task | Held-in input, full-channel reconstruction + forward prediction |

STNDT-lite headline (`0.3830 co-bps`) is **5 ms**; LFADS upstream MC_Maze example is **20 ms**.

## Smoke overrides

See `mc_maze_20ms_smoke.yaml` and `smoke_single.yaml` (Hydra root for smoke/export).

Workflow after smoke train:

1. `scripts/export_lfads_rates.py --run-dir results/lfads_smoke/<run_id>`
2. `scripts/evaluate_lfads_outputs.py --run-dir results/lfads_smoke/<run_id>`

NLB keys at 20 ms: `mc_maze_20` / `mc_maze_20_split`.
