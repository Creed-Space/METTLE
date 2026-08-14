"""Tests for the public distribution receipt."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from scripts.build_distribution_receipt import build_receipt
from scripts.verify_pypi_release import (
    MCP_SERVER_NAME,
    _RejectRedirects,
    _load_reproducibility_receipt,
    _require_https,
    _require_mcp_ownership_marker,
    _version_json_url,
)


def test_mcp_ownership_marker_matches_registry_identity() -> None:
    marker = f"mcp-name: {MCP_SERVER_NAME}"
    assert _require_mcp_ownership_marker(f"<!-- {marker} -->") == marker


def test_mcp_ownership_marker_rejects_missing_or_wrong_identity() -> None:
    with pytest.raises(RuntimeError, match="ownership marker"):
        _require_mcp_ownership_marker("mcp-name: io.github.example/other")


def test_publication_verifier_targets_one_exact_version() -> None:
    assert _version_json_url("0.4.0") == (
        "https://pypi.org/pypi/mettle-verifier/0.4.0/json"
    )
    with pytest.raises(ValueError, match="unsafe PyPI version"):
        _version_json_url("../forged")


def test_publication_verifier_loads_source_bound_reproducible_hashes(
    tmp_path: Path,
) -> None:
    _pypi, reproducibility, _registry = _inputs(tmp_path)

    assert _load_reproducibility_receipt(
        reproducibility,
        version="0.3.1",
        source_sha="c" * 40,
    ) == {
        "mettle_verifier-0.3.1-py3-none-any.whl": "a" * 64,
        "mettle_verifier-0.3.1.tar.gz": "b" * 64,
    }


@pytest.mark.parametrize("mutation", ["source", "result", "duplicate", "digest"])
def test_publication_verifier_rejects_untrusted_reproducibility_receipts(
    tmp_path: Path,
    mutation: str,
) -> None:
    _pypi, reproducibility, _registry = _inputs(tmp_path)
    payload = json.loads(reproducibility.read_text())
    if mutation == "source":
        payload["source_sha"] = "d" * 40
    elif mutation == "result":
        payload["result"] = "different"
    elif mutation == "duplicate":
        payload["artifacts"][1] = payload["artifacts"][0]
    else:
        payload["artifacts"][0]["sha256"] = "not-a-digest"
    reproducibility.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="reproducibility"):
        _load_reproducibility_receipt(
            reproducibility,
            version="0.3.1",
            source_sha="c" * 40,
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    filename_hashes = {
        "mettle_verifier-0.3.1-py3-none-any.whl": "a" * 64,
        "mettle_verifier-0.3.1.tar.gz": "b" * 64,
    }
    pypi = tmp_path / "pypi.json"
    pypi.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "version": "0.3.1",
                "artifacts": [
                    {"filename": name, "sha256": digest}
                    for name, digest in filename_hashes.items()
                ],
            }
        )
    )
    reproducibility = tmp_path / "repro.json"
    reproducibility.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": "byte-identical",
                "source_sha": "c" * 40,
                "artifacts": [
                    {"name": name, "sha256": digest}
                    for name, digest in filename_hashes.items()
                ],
            }
        )
    )
    registry = {
        "server": {
            "name": "io.github.Creed-Space/mettle-mcp",
            "version": "0.3.1",
            "packages": [
                {
                    "registryType": "pypi",
                    "identifier": "mettle-verifier",
                    "version": "0.3.1",
                }
            ],
        },
        "_meta": {
            "io.modelcontextprotocol.registry/official": {
                "status": "active",
                "isLatest": True,
            }
        },
    }
    return pypi, reproducibility, registry


def test_distribution_receipt_binds_all_public_identities(tmp_path: Path) -> None:
    pypi, reproducibility, registry = _inputs(tmp_path)
    receipt = build_receipt(
        pypi_path=pypi,
        reproducibility_path=reproducibility,
        source_sha="c" * 40,
        tag="v0.3.1",
        workflow_url="https://github.example/run/1",
        registry=registry,
    )
    assert receipt["tag"] == "v0.3.1"
    assert receipt["official_mcp_registry"] == registry


def test_distribution_receipt_rejects_registry_drift(tmp_path: Path) -> None:
    pypi, reproducibility, registry = _inputs(tmp_path)
    registry["server"]["version"] = "0.3.0"
    with pytest.raises(ValueError, match="Registry"):
        build_receipt(
            pypi_path=pypi,
            reproducibility_path=reproducibility,
            source_sha="c" * 40,
            tag="v0.3.1",
            workflow_url="https://github.example/run/1",
            registry=registry,
        )


def test_distribution_receipt_rejects_duplicate_artifact_names(tmp_path: Path) -> None:
    pypi, reproducibility, registry = _inputs(tmp_path)
    payload = json.loads(pypi.read_text())
    payload["artifacts"].append(payload["artifacts"][0])
    pypi.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate artifact"):
        build_receipt(
            pypi_path=pypi,
            reproducibility_path=reproducibility,
            source_sha="c" * 40,
            tag="v0.3.1",
            workflow_url="https://github.example/run/1",
            registry=registry,
        )


def test_publication_downloads_require_exact_https_origins() -> None:
    _require_https("https://pypi.org/pypi/mettle-verifier/json", {"pypi.org"})
    with pytest.raises(ValueError, match="untrusted publication URL"):
        _require_https("file:///tmp/forged-wheel", {"files.pythonhosted.org"})
    with pytest.raises(ValueError, match="untrusted publication URL"):
        _require_https(
            "https://files.pythonhosted.org.example/file", {"files.pythonhosted.org"}
        )


def test_publication_verifier_rejects_redirects_before_following() -> None:
    with pytest.raises(urllib.error.HTTPError, match="redirect rejected"):
        _RejectRedirects().redirect_request(
            SimpleNamespace(full_url="https://pypi.org/source"),
            None,
            302,
            "Found",
            {},
            "https://attacker.example/target",
        )
