"""Deployment dependency contract tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_requirements_include_challenge_engine_runtime() -> None:
    """The hosted v2 route imports NumPy through ``scripts.engine``."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
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


def test_mcp2_contract_and_hashed_container_lock() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-mcp.txt").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-mcp-lock.txt").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert '"mcp>=2.0,<3"' in pyproject
    assert "mcp>=2.0,<3" in requirements
    assert "mcp==2.0.0" in lock
    assert lock.count("--hash=sha256:") >= 10
    assert "--require-hashes -r requirements-mcp-lock.txt" in dockerfile
    assert "pip install --require-hashes -r requirements-mcp-lock.txt" in ci
    assert "pip install --no-cache-dir --no-deps ." in dockerfile


def test_mcp_container_context_and_entrypoint_are_bounded() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert dockerignore.splitlines()[2] == "**"
    assert "!mettle/**" in dockerignore
    assert "!deploy/mcp/entrypoint.sh" in dockerignore
    assert "!static/" not in dockerignore

    entrypoint = ROOT / "deploy/mcp/entrypoint.sh"
    result = subprocess.run(
        ["sh", str(entrypoint)],
        env={**os.environ, "METTLE_MCP_TRANSPORT": "invalid"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert result.stderr == "Unsupported METTLE_MCP_TRANSPORT: invalid\n"


def test_render_auto_deploy_is_the_only_main_deployment_authority() -> None:
    """The hotfix must not race Render auto-deploy through a second hook."""
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "autoDeploy: true" in render
    assert not (ROOT / ".github/workflows/deploy.yml").exists()
