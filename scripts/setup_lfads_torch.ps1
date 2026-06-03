# Setup isolated LFADS environment (does not modify STNDT-lite nlb env).
# Requires git. Conda recommended; see docs/lfads_baseline_plan.md for manual steps.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LfadsDir = if ($env:LFADS_TORCH_DIR) { $env:LFADS_TORCH_DIR } else { Join-Path $RepoRoot "external\lfads-torch" }
$EnvName = if ($env:LFADS_CONDA_ENV) { $env:LFADS_CONDA_ENV } else { "lfads-nlb" }

Write-Host "Repo: $RepoRoot"
Write-Host "lfads-torch target: $LfadsDir"

if (-not (Test-Path $LfadsDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LfadsDir) | Out-Null
    git clone --depth 1 https://github.com/arsedler9/lfads-torch.git $LfadsDir
} else {
    Write-Host "lfads-torch directory exists; run 'git -C $LfadsDir pull' to update if needed."
}

function Test-Conda {
    return [bool](Get-Command conda -ErrorAction SilentlyContinue)
}

if (Test-Conda) {
    Write-Host "Creating conda env '$EnvName' (python=3.9) if missing..."
    conda create -n $EnvName python=3.9 -y 2>$null | Out-Null
    Write-Host @"

Activate and install (manual — conda activate from scripts is unreliable):

  conda activate $EnvName
  pip install --upgrade pip
  pip install -e "$LfadsDir" --no-deps
  pip install torch==1.13.1 pytorch-lightning==1.6.0 torchmetrics==0.7.2 hydra-core==1.3.0 h5py "numpy<2" scikit-learn matplotlib
  pip install "pandas==1.3.4" nlb-tools==0.0.4
  rem ray[tune] only needed for PBT/multi; optional: pip install "ray[tune]>=2.2,<3"

"@
} else {
    Write-Host "conda not found. Use Python 3.9/3.10 venv instead (see docs/lfads_baseline_plan.md)."
    $VenvDir = Join-Path $RepoRoot ".venv-lfads-nlb"
    if (-not (Test-Path $VenvDir)) {
        py -3.10 -m venv $VenvDir
    }
    Write-Host @"

  $VenvDir\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -e "$LfadsDir" --no-deps
  pip install torch==1.13.1 pytorch-lightning==1.6.0 torchmetrics==0.7.2 hydra-core==1.3.0 h5py "numpy<2" scikit-learn matplotlib
  pip install "pandas==1.3.4" nlb-tools==0.0.4
  rem ray[tune] only needed for PBT/multi; optional: pip install "ray[tune]>=2.2,<3"

"@
}

Write-Host "Verify imports:"
Write-Host "  python -c `"import torch; import lfads_torch; import nlb_tools; print('ok')`""
Write-Host ""
Write-Host "Prepare data:"
Write-Host "  python scripts/prepare_lfads_mc_maze.py --write-smoke-subset"
Write-Host "Smoke train:"
Write-Host "  python scripts/run_lfads_mc_maze_smoke.py"
