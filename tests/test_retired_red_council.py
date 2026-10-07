"""Retired demonstration entrypoints must remain offline and report their limits."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "runner",
    ["red_team/run_red_council.py", "scripts/testing/run_mettle_red_council.py"],
)
def test_retired_runner_emits_no_measurement(runner, tmp_path):
    output = tmp_path / "retirement.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / runner), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    notice = json.loads(output.read_text())
    assert notice["status"] == "retired"
    assert notice["scenarios_executed"] == 0
    assert notice["evaluation_mode"] == "fixed_demonstration"
    assert "not a measurement" in notice["evaluation_caveat"]
    assert notice["evaluation_caveat"] in result.stdout


@pytest.mark.parametrize(
    "runner",
    ["red_team/run_red_council.py", "scripts/testing/run_mettle_red_council.py"],
)
def test_retired_runner_imports_only_standard_library(runner):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / runner).read_text())
    allowed = {"__future__", "argparse", "json", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed
