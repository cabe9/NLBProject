"""Smoke test: importing the top-level package must not raise."""


def test_import_pipeline():
    """The ``nlb_project`` package imports cleanly with no side effects."""
    import nlb_project  # noqa: F401
