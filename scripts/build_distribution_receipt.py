"""Bind PyPI, Official MCP Registry, and reproducibility evidence."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REGISTRY_BASE = "https://registry.modelcontextprotocol.io/v0.1/servers"
SERVER_NAME = "io.github.Creed-Space/mettle-mcp"


def _registry_latest() -> dict[str, Any]:
    encoded = urllib.parse.quote(SERVER_NAME, safe="")
    request = urllib.request.Request(
        f"{REGISTRY_BASE}/{encoded}/versions/latest",
        headers={"User-Agent": "METTLE-Distribution-Receipt/1.0"},
    )
    # The URL is constructed from a fixed HTTPS registry origin and quoted name.
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Official MCP Registry returned a non-object response")
    return payload


def wait_for_registry(version: str, *, attempts: int, delay: float) -> dict[str, Any]:
    """Wait for the Official MCP Registry's latest pointer to converge."""
    last_version = "unavailable"
    for attempt in range(1, attempts + 1):
        try:
            payload = _registry_latest()
        except urllib.error.URLError:
            payload = {}
            last_version = "network-error"
        last_version = str(payload.get("server", {}).get("version", "missing"))
        official = payload.get("_meta", {}).get(
            "io.modelcontextprotocol.registry/official", {}
        )
        if (
            last_version == version
            and official.get("status") == "active"
            and official.get("isLatest") is True
        ):
            return payload
        if attempt < attempts:
            time.sleep(delay)
    raise RuntimeError(
        f"Official MCP Registry latest did not converge to {version}; "
        f"last observed {last_version}"
    )


def build_receipt(
    *,
    pypi_path: Path,
    reproducibility_path: Path,
    source_sha: str,
    tag: str,
    workflow_url: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and combine all public distribution identities."""
    pypi = json.loads(pypi_path.read_text(encoding="utf-8"))
    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    registry_payload = registry if registry is not None else _registry_latest()
    version = pypi["version"]
    if tag != f"v{version}":
        raise ValueError(f"tag {tag} does not match PyPI version {version}")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA must be a full lowercase Git SHA")
    if reproducibility["source_sha"] != source_sha:
        raise ValueError("source SHA does not match reproducibility evidence")
    if (
        pypi.get("schema_version") != "1.0"
        or reproducibility.get("schema_version") != "1.0"
    ):
        raise ValueError("distribution evidence uses an unsupported schema")
    if reproducibility["result"] != "byte-identical":
        raise ValueError("distribution build is not byte reproducible")

    server = registry_payload["server"]
    official = registry_payload["_meta"]["io.modelcontextprotocol.registry/official"]
    if len(server.get("packages", [])) != 1:
        raise ValueError("Official MCP Registry package list is ambiguous")
    package = server["packages"][0]
    if (
        server["name"] != SERVER_NAME
        or server["version"] != version
        or package["registryType"] != "pypi"
        or package["identifier"] != "mettle-verifier"
        or package["version"] != version
        or official["status"] != "active"
        or official["isLatest"] is not True
    ):
        raise ValueError("Official MCP Registry does not expose the expected release")

    pypi_artifacts = pypi["artifacts"]
    reproducible_artifacts = reproducibility["artifacts"]
    pypi_hashes = {
        artifact["filename"]: artifact["sha256"] for artifact in pypi["artifacts"]
    }
    reproducible_hashes = {
        artifact["name"]: artifact["sha256"] for artifact in reproducible_artifacts
    }
    if len(pypi_hashes) != len(pypi_artifacts) or len(reproducible_hashes) != len(
        reproducible_artifacts
    ):
        raise ValueError("distribution evidence contains duplicate artifact names")
    if len(pypi_hashes) != 2:
        raise ValueError("distribution evidence must contain one wheel and one sdist")
    if pypi_hashes != reproducible_hashes:
        raise ValueError("PyPI artifacts differ from independent reproducible builds")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_sha": source_sha,
        "tag": tag,
        "workflow_url": workflow_url,
        "pypi": pypi,
        "official_mcp_registry": registry_payload,
        "reproducibility": reproducibility,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypi", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--attempts", type=int, default=18)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        pypi_path=args.pypi,
        reproducibility_path=args.reproducibility,
        source_sha=args.source_sha,
        tag=args.tag,
        workflow_url=args.workflow_url,
        registry=wait_for_registry(
            args.tag.removeprefix("v"), attempts=args.attempts, delay=args.delay
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
