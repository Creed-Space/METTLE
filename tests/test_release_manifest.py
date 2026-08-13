"""Release manifests bind protocol metadata, notes, artifacts, and source SHA."""

from pathlib import Path

import pytest

from scripts.build_release_manifest import build_manifest, load_protocol_versions


def test_release_manifest_reads_protocol_versions_without_runtime_imports(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.py"
    protocol.write_text(
        'CREDENTIAL_SCHEMA_VERSION = "1.0"\nSUITE_POLICY_VERSION = "policy-1"\n',
        encoding="utf-8",
    )
    assert load_protocol_versions(protocol) == ("1.0", "policy-1")

    protocol.write_text('CREDENTIAL_SCHEMA_VERSION = "1.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="constants are missing"):
        load_protocol_versions(protocol)


def test_release_manifest_records_artifact_hashes_and_policy(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "mettle.whl").write_bytes(b"wheel")
    output = dist / "RELEASE-MANIFEST.json"
    body = dist / "RELEASE-NOTES.md"

    manifest = build_manifest(
        dist=dist,
        notes_path=Path("RELEASE_NOTES.md"),
        source_sha="a" * 40,
        tag="v0.3.2",
        output=output,
        release_body=body,
    )

    assert manifest["source_sha"] == "a" * 40
    assert manifest["credential_schema_version"] == "1.0"
    assert manifest["suite_policy_version"] == "2026-08-12"
    assert manifest["artifacts"][0]["name"] == "mettle.whl"
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert output.is_file()
    assert body.read_text(encoding="utf-8").startswith("## [0.3.2]")


def test_release_manifest_rejects_tag_or_sha_drift(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "artifact").write_text("x", encoding="utf-8")
    common = {
        "dist": dist,
        "notes_path": Path("RELEASE_NOTES.md"),
        "output": dist / "manifest.json",
        "release_body": dist / "notes.md",
    }

    with pytest.raises(ValueError, match="does not match"):
        build_manifest(source_sha="a" * 40, tag="v9.9.9", **common)
    with pytest.raises(ValueError, match="source SHA"):
        build_manifest(source_sha="short", tag="v0.3.2", **common)
