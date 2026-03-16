from __future__ import annotations

from pathlib import Path

import pytest

from nlb_project.data_contract import resolve_data_path


def test_resolve_data_path_from_explicit_path(tmp_path: Path) -> None:
    ds_dir = tmp_path / "mc_maze"
    ds_dir.mkdir(parents=True)
    (ds_dir / "session_full_a.nwb").write_text("", encoding="utf-8")

    out = resolve_data_path("mc_maze", str(ds_dir), "*full")
    assert out == str(ds_dir.resolve())


def test_resolve_data_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "raw"
    ds_dir = root / "000128" / "sub-Jenkins"
    ds_dir.mkdir(parents=True)
    (ds_dir / "session_full_a.nwb").write_text("", encoding="utf-8")

    monkeypatch.setenv("NLB_DATA_DIR", str(root))
    out = resolve_data_path("mc_maze", None, "*full")
    assert out == str(ds_dir.resolve())


def test_resolve_data_path_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLB_DATA_DIR", raising=False)
    with pytest.raises(ValueError):
        resolve_data_path("mc_maze", None, "*full")
