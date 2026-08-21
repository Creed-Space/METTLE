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


def test_mcp_requirements_pin_the_compatible_major_line() -> None:
    """The MCP adapter uses the security-maintained v1 decorator contract."""
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements-mcp.txt"
    ).read_text(encoding="utf-8")
    normalized = {
        line.split("#", 1)[0].strip().lower().replace(" ", "")
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    }
    assert "mcp>=1.28,<2" in normalized
