#!/usr/bin/env bash
# Sync Mac NLBProject from PC-validated Git state.
# Prefer GitHub branch: pc/validated-screen-c-rtx3080
# Offline fallback: git bundle (see scripts/SYNC_PC_TO_MAC.md)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PC_BRANCH="pc/validated-screen-c-rtx3080"
BACKUP_BRANCH="mac-backup-before-pc-sync"

echo "== Mac repo: $(pwd) =="
echo "Current branch: $(git branch --show-current)"
git status --short

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Uncommitted Mac changes detected — creating backup branch."
  git checkout -b "$BACKUP_BRANCH" 2>/dev/null || git checkout "$BACKUP_BRANCH"
  git add -A
  git -c user.name="${GIT_USER_NAME:-Mac Backup}" \
      -c user.email="${GIT_USER_EMAIL:-mac-backup@local}" \
      commit -m "Backup Mac state before syncing from PC" || true
fi

git fetch origin "$PC_BRANCH" || {
  echo "Fetch failed. If using a bundle:"
  echo "  git fetch /path/to/NLBProject-pc-validated-screen-c.bundle $PC_BRANCH:$PC_BRANCH"
  exit 1
}

TARGET_BRANCH="${MAC_SYNC_TARGET_BRANCH:-master}"
git checkout "$TARGET_BRANCH"
git merge --no-ff "origin/$PC_BRANCH" -m "Merge PC validated Screen C state ($PC_BRANCH)"

echo "Sync complete. Run checks:"
echo "  ruff format --check ."
echo "  ruff check ."
echo "  python -m pytest -q tests/test_config.py tests/test_pipeline_full_val_selection.py"
