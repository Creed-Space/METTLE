"""Deployment dependency contract tests."""

from __future__ import annotations

from pathlib import Path


def test_server_requirements_include_challenge_engine_runtime() -> None:
    """The hosted v2 route imports NumPy through ``scripts.engine``."""
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    normalized = {
        line.split("#", 1)[0].strip().lower()
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    }
    assert any(
        requirement == "numpy"
        or requirement.startswith(("numpy=", "numpy<", "numpy>", "numpy!", "numpy~"))
        for requirement in normalized
    )
