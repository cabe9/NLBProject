from __future__ import annotations

from pathlib import Path

from nlb_project.public_test import download_public_test_eval_data, read_public_metrics, sha256_file


def test_download_public_test_eval_data_accepts_existing_verified_file(tmp_path: Path) -> None:
    target = tmp_path / "eval_data_test.h5"
    target.write_bytes(b"fixture")
    expected_sha = sha256_file(target)

    result = download_public_test_eval_data(
        target,
        expected_sha256=expected_sha,
    )

    assert result.path == target
    assert result.sha256 == expected_sha
    assert result.size_bytes == len(b"fixture")
    assert not result.downloaded


def test_download_public_test_eval_data_rejects_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "eval_data_test.h5"
    target.write_bytes(b"stale")

    try:
        download_public_test_eval_data(target, expected_sha256="not-the-sha")
    except ValueError as exc:
        assert "exists but has sha256" in str(exc)
    else:
        raise AssertionError("expected stale public eval data to be rejected")


def test_read_public_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("model,co-bps\nselected,0.1\n", encoding="utf-8")

    assert read_public_metrics(path) == [{"model": "selected", "co-bps": "0.1"}]
