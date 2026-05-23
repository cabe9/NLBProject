# Sync PC (RTX 3080) → Mac

PC validated state lives on branch **`pc/validated-screen-c-rtx3080`** (commit includes Screen C
infrastructure, public-test configs/docs, `pipeline.py` full-val selection, etc.).

## Option A — GitHub (preferred)

### On PC (once)

```powershell
cd C:\Users\david\NLBProject
git push -u origin pc/validated-screen-c-rtx3080
```

Authenticate with GitHub (browser or PAT) when prompted.

### On Mac

```bash
cd /path/to/NLBProject
git status
git branch --show-current

# Backup uncommitted Mac work
git checkout -b mac-backup-before-pc-sync
git add .
git commit -m "Backup Mac state before syncing from PC"

git fetch origin pc/validated-screen-c-rtx3080
git checkout master   # or your usual branch
git merge --no-ff origin/pc/validated-screen-c-rtx3080 -m "Merge PC validated Screen C state"

# Or run: ./scripts/mac-sync-from-pc.sh
```

## Option B — Git bundle (no GitHub push)

### On PC

```powershell
cd C:\Users\david\NLBProject
git bundle create ..\NLBProject-pc-validated-screen-c.bundle pc/validated-screen-c-rtx3080
```

Copy `NLBProject-pc-validated-screen-c.bundle` to the Mac (AirDrop, cloud, USB).

### On Mac

```bash
cd /path/to/NLBProject
git fetch /path/to/NLBProject-pc-validated-screen-c.bundle pc/validated-screen-c-rtx3080:pc/validated-screen-c-rtx3080
git checkout master
git merge --no-ff pc/validated-screen-c-rtx3080
```

## Intentionally not synced

- `data/` (raw NWB, eval HDF5)
- `.venv/` / virtualenvs
- `results/**/predictions/`, `run_metadata.json`, `results/public_test/`
- `*.log`, `.cursor/`, caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`)
- Machine-specific run logs at repo root

Only `results/benchmark_runs/*/metrics.csv` and portfolio comparison files are tracked per `.gitignore`.

## Post-sync checks (Mac)

```bash
ruff format --check .
ruff check .
python -m pytest -q tests/test_config.py tests/test_pipeline_full_val_selection.py
```
