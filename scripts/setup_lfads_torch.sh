#!/usr/bin/env bash
# Setup isolated LFADS environment (does not modify STNDT-lite nlb env).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LFADS_DIR="${LFADS_TORCH_DIR:-$REPO_ROOT/external/lfads-torch}"
ENV_NAME="${LFADS_CONDA_ENV:-lfads-nlb}"

echo "Repo: $REPO_ROOT"
echo "lfads-torch target: $LFADS_DIR"

if [[ ! -d "$LFADS_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LFADS_DIR")"
  git clone --depth 1 https://github.com/arsedler9/lfads-torch.git "$LFADS_DIR"
else
  echo "lfads-torch exists; git -C $LFADS_DIR pull to update"
fi

if command -v conda >/dev/null 2>&1; then
  conda create -n "$ENV_NAME" python=3.9 -y 2>/dev/null || true
  cat <<EOF

Activate and install (run after: conda activate $ENV_NAME):

  pip install --upgrade pip
  pip install -e "$LFADS_DIR" --no-deps
  pip install torch==1.13.1 pytorch-lightning==1.6.0 torchmetrics==0.7.2 hydra-core==1.3.0 h5py 'numpy<2' scikit-learn matplotlib
  pip install 'pandas==1.3.4' nlb-tools==0.0.4
  # ray[tune] only for PBT/multi-session; smoke uses a stub for single-session runs

EOF
else
  VENV_DIR="$REPO_ROOT/.venv-lfads-nlb"
  python3.10 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
  cat <<EOF

  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -e "$LFADS_DIR" --no-deps
  pip install torch==1.13.1 pytorch-lightning==1.6.0 torchmetrics==0.7.2 hydra-core==1.3.0 h5py 'numpy<2' scikit-learn matplotlib
  pip install 'pandas==1.3.4' nlb-tools==0.0.4
  # ray[tune] only for PBT/multi-session; smoke uses a stub for single-session runs

EOF
fi

echo "Verify: python -c 'import torch; import lfads_torch; import nlb_tools; print(\"ok\")'"
echo "Data:    python scripts/prepare_lfads_mc_maze.py --write-smoke-subset"
echo "Smoke:   python scripts/run_lfads_mc_maze_smoke.py"
