#!/usr/bin/env python3
"""Build a SHA-bound, version-aware release manifest and curated release body."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mettle.protocol import CREDENTIAL_SCHEMA_VERSION, SUITE_POLICY_VERSION  # noqa: E402

REQUIRED_SECTIONS = (
    "Credential schema",
    "Suite policy",
    "Public key changes",
    "Compatibility",
    "Known limitations",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_section(notes: str, version: str) -> tuple[str, dict[str, str]]:
    heading = re.compile(rf"^## \[{re.escape(version)}\]\s*$", re.MULTILINE)
    match = heading.search(notes)
    if match is None:
        raise ValueError(f"release notes do not contain version {version}")
    next_release = re.search(r"^## \[", notes[match.end() :], re.MULTILINE)
    end = match.end() + next_release.start() if next_release else len(notes)
    rendered = notes[match.start() : end].strip() + "\n"
    sections: dict[str, str] = {}
    for title in REQUIRED_SECTIONS:
        section_match = re.search(
            rf"^### {re.escape(title)}\s*$\n(.*?)(?=^### |\Z)",
            rendered,
            re.MULTILINE | re.DOTALL,
        )
        if section_match is None or not section_match.group(1).strip():
            raise ValueError(f"release notes section {title!r} is missing or empty")
        sections[title] = section_match.group(1).strip()
    return rendered, sections


def build_manifest(
    *,
    dist: Path,
    notes_path: Path,
    source_sha: str,
    tag: str,
    output: Path,
    release_body: Path,
) -> dict[str, Any]:
    if not dist.is_dir():
        raise ValueError(f"release artifact directory does not exist: {dist}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match package version v{version}")
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", source_sha) is None:
        raise ValueError("source SHA must be a full 40 or 64 character lowercase hash")

    notes_text = notes_path.read_text(encoding="utf-8")
    rendered_notes, sections = release_section(notes_text, version)
    release_body.write_text(rendered_notes, encoding="utf-8")

    excluded = {output.resolve(), release_body.resolve()}
    artifacts = []
    for artifact in sorted(path for path in dist.iterdir() if path.is_file()):
        if artifact.resolve() in excluded or artifact.name == "SHA256SUMS":
            continue
        artifacts.append(
            {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
        )
    if not artifacts:
        raise ValueError("release manifest requires at least one built artifact")

    manifest = {
        "manifest_schema_version": "1.0",
        "source_sha": source_sha,
        "tag": tag,
        "package_version": version,
        "credential_schema_version": CREDENTIAL_SCHEMA_VERSION,
        "suite_policy_version": SUITE_POLICY_VERSION,
        "public_key_changes": sections["Public key changes"],
        "compatibility": sections["Compatibility"],
        "known_limitations": sections["Known limitations"],
        "release_notes_sha256": hashlib.sha256(
            rendered_notes.encode("utf-8")
        ).hexdigest(),
        "artifacts": artifacts,
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--notes", type=Path, default=ROOT / "RELEASE_NOTES.md")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release-body", type=Path)
    args = parser.parse_args()
    output = args.output or args.dist / "RELEASE-MANIFEST.json"
    release_body = args.release_body or args.dist / "RELEASE-NOTES.md"
    manifest = build_manifest(
        dist=args.dist,
        notes_path=args.notes,
        source_sha=args.source_sha,
        tag=args.tag,
        output=output,
        release_body=release_body,
    )
    print(
        f"Release manifest covers {len(manifest['artifacts'])} artifacts at "
        f"{manifest['source_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
