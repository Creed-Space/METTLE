"""Tests for the reproducible hosted-server SBOM finalization step."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.finalize_server_sbom import finalize_bom


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    lock = tmp_path / "requirements-production.txt"
    lock.write_text("fastapi[standard]==1.0\npydantic==2.0\n", encoding="utf-8")
    direct = tmp_path / "requirements.txt"
    direct.write_text("fastapi>=1.0\npydantic>=2.0  # models\n", encoding="utf-8")
    return lock, direct


def _bom() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {"component": {"name": "mettle", "bom-ref": "root"}},
        "components": [
            {"name": "fastapi", "version": "1.0", "bom-ref": "fastapi==1.0"},
            {"name": "pydantic", "version": "2.0", "bom-ref": "pydantic==2.0"},
            {"name": "pip", "version": "99", "bom-ref": "pip==99"},
        ],
        "dependencies": [
            {"ref": "root", "dependsOn": ["pydantic==2.0"]},
            {"ref": "fastapi==1.0", "dependsOn": ["pydantic==2.0"]},
            {"ref": "pydantic==2.0"},
            {"ref": "pip==99"},
        ],
    }


def test_finalize_server_bom_filters_tools_and_links_direct_dependencies(
    tmp_path: Path,
) -> None:
    lock, direct = _source_files(tmp_path)

    result = finalize_bom(copy.deepcopy(_bom()), lock_path=lock, direct_path=direct)

    assert [component["name"] for component in result["components"]] == [
        "fastapi",
        "pydantic",
    ]
    graph = {
        entry["ref"]: entry.get("dependsOn", []) for entry in result["dependencies"]
    }
    assert graph["root"] == ["fastapi==1.0", "pydantic==2.0"]
    assert graph["fastapi==1.0"] == ["pydantic==2.0"]
    assert "pip==99" not in graph


def test_finalize_server_bom_rejects_missing_locked_component(tmp_path: Path) -> None:
    lock, direct = _source_files(tmp_path)
    incomplete = _bom()
    incomplete["components"] = incomplete["components"][:1]

    with pytest.raises(ValueError, match="omitted locked components"):
        finalize_bom(incomplete, lock_path=lock, direct_path=direct)
