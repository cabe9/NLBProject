"""Tests for data-path resolution: explicit path, env var, and error path.

These pin the "how does the pipeline find raw NWB data?" contract so that
local dev (explicit ``data_path``) and CI (``NLB_DATA_DIR`` env var) both
work, and so that misconfiguration fails with a readable ``ValueError``
rather than an opaque IO error later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlb_project.data_contract import resolve_data_path


def test_resolve_data_path_from_explicit_path(tmp_path: Path) -> None:
    """An explicit ``data_path`` resolves to that path verbatim."""
    ds_dir = tmp_path / "mc_maze"
    ds_dir.mkdir(parents=True)
    (ds_dir / "session_full_a.nwb").write_text("", encoding="utf-8")

    out = resolve_data_path("mc_maze", str(ds_dir), "*full")
    assert out == str(ds_dir.resolve())


def test_resolve_data_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no explicit path, ``NLB_DATA_DIR`` is used to locate the dataset."""
    root = tmp_path / "raw"
    ds_dir = root / "000128" / "sub-Jenkins"
    ds_dir.mkdir(parents=True)
    (ds_dir / "session_full_a.nwb").write_text("", encoding="utf-8")

    monkeypatch.setenv("NLB_DATA_DIR", str(root))
    out = resolve_data_path("mc_maze", None, "*full")
    assert out == str(ds_dir.resolve())


def test_resolve_data_path_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing both explicit path and env var raises a clear ``ValueError``."""
    monkeypatch.delenv("NLB_DATA_DIR", raising=False)
    with pytest.raises(ValueError):
        resolve_data_path("mc_maze", None, "*full")
