#!/usr/bin/env python3
"""Finalize a reproducible CycloneDX graph for the hosted METTLE server.

``cyclonedx-py environment`` can recover transitive dependency edges from an
installed environment, but a PEP 621 root only knows the wheel's deliberately
small CLI dependency set. The hosted server's direct dependencies live in
``requirements.txt``. This step joins those two authoritative inputs, removes
venv bootstrap tools that are absent from the production lock, and rejects any
dangling or missing graph reference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LOCKED_NAME = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==")
DIRECT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(?:[<>=!~].*)?$")


def canonical_name(value: str) -> str:
    """Return the normalized distribution name defined by PEP 503."""
    return re.sub(r"[-_.]+", "-", value).lower()


def locked_names(path: Path) -> set[str]:
    """Read every exact distribution name from a pip-compile lock."""
    names = {
        canonical_name(match.group(1))
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (match := LOCKED_NAME.match(raw.strip())) is not None
    }
    if not names:
        raise ValueError(f"production lock contains no exact requirements: {path}")
    return names


def direct_names(path: Path) -> set[str]:
    """Read top-level server dependencies from the human-maintained source."""
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        requirement = raw.split("#", 1)[0].strip()
        if not requirement:
            continue
        match = DIRECT_NAME.fullmatch(requirement)
        if match is None:
            raise ValueError(f"unsupported direct requirement syntax: {requirement!r}")
        names.add(canonical_name(match.group(1)))
    if not names:
        raise ValueError(f"direct requirement source is empty: {path}")
    return names


def finalize_bom(
    bom: dict[str, Any], *, lock_path: Path, direct_path: Path
) -> dict[str, Any]:
    """Return a filtered, root-linked, internally consistent CycloneDX BOM."""
    lock = locked_names(lock_path)
    direct = direct_names(direct_path)
    if not direct <= lock:
        missing = sorted(direct - lock)
        raise ValueError(f"direct requirements absent from production lock: {missing}")

    metadata = bom.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    root_ref = root.get("bom-ref") if isinstance(root, dict) else None
    if not isinstance(root_ref, str) or not root_ref:
        raise ValueError("SBOM metadata is missing a root component bom-ref")

    components = bom.get("components")
    if not isinstance(components, list):
        raise ValueError("SBOM components must be a list")
    filtered = [
        component
        for component in components
        if isinstance(component, dict)
        and canonical_name(str(component.get("name", ""))) in lock
    ]
    by_name: dict[str, str] = {}
    for component in filtered:
        name = canonical_name(str(component.get("name", "")))
        reference = component.get("bom-ref")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"locked component {name!r} has no bom-ref")
        if name in by_name:
            raise ValueError(f"SBOM contains duplicate locked component {name!r}")
        by_name[name] = reference
    missing_components = sorted(lock - by_name.keys())
    if missing_components:
        raise ValueError(
            f"installed environment omitted locked components: {missing_components}"
        )

    allowed_refs = set(by_name.values()) | {root_ref}
    dependencies = bom.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("SBOM dependencies must be a list")
    graph: dict[str, set[str]] = {reference: set() for reference in allowed_refs}
    for entry in dependencies:
        if not isinstance(entry, dict) or entry.get("ref") not in allowed_refs:
            continue
        reference = str(entry["ref"])
        children = entry.get("dependsOn", [])
        if not isinstance(children, list):
            raise ValueError(f"dependency entry {reference!r} has invalid dependsOn")
        graph[reference].update(
            child
            for child in children
            if isinstance(child, str) and child in allowed_refs
        )
    graph[root_ref].update(by_name[name] for name in direct)

    bom["components"] = sorted(
        filtered,
        key=lambda component: (
            canonical_name(str(component.get("name", ""))),
            str(component.get("version", "")),
            str(component.get("bom-ref", "")),
        ),
    )
    bom["dependencies"] = [
        {"ref": reference, "dependsOn": sorted(graph[reference])}
        if graph[reference]
        else {"ref": reference}
        for reference in sorted(graph)
    ]

    emitted_refs = {entry["ref"] for entry in bom["dependencies"]}
    dangling = {
        child
        for entry in bom["dependencies"]
        for child in entry.get("dependsOn", [])
        if child not in emitted_refs
    }
    if dangling:
        raise ValueError(
            f"SBOM dependency graph has dangling references: {sorted(dangling)}"
        )
    return bom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=Path("requirements-production.txt")
    )
    parser.add_argument("--direct", type=Path, default=Path("requirements.txt"))
    args = parser.parse_args()

    bom = json.loads(args.input.read_text(encoding="utf-8"))
    finalized = finalize_bom(bom, lock_path=args.lock, direct_path=args.direct)
    args.output.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    root_ref = finalized["metadata"]["component"]["bom-ref"]
    root_entry = next(
        entry for entry in finalized["dependencies"] if entry["ref"] == root_ref
    )
    print(
        f"Server SBOM finalized: {len(finalized['components'])} locked components, "
        f"{len(root_entry.get('dependsOn', []))} direct dependencies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
