"""Verify one public PyPI release and emit a hash-bound receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path
from typing import Any


PYPI_JSON = "https://pypi.org/pypi/mettle-verifier/json"
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


def _require_https(url: str, hosts: set[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise ValueError(
            f"untrusted publication URL origin: {parsed.scheme}://{parsed.hostname}"
        )


def _get_json(url: str) -> dict[str, Any]:
    _require_https(url, {"pypi.org"})
    request = urllib.request.Request(
        url, headers={"User-Agent": "METTLE-Publication-Verification/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("PyPI returned a non-object response")
    return payload


def _download(url: str, target: Path, expected_size: int) -> str:
    _require_https(url, {"files.pythonhosted.org"})
    if expected_size < 1 or expected_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"untrusted publication size: {expected_size}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "METTLE-Publication-Verification/1.0"}
    )
    digest = hashlib.sha256()
    observed_size = 0
    with (
        urllib.request.urlopen(request, timeout=60) as response,  # nosec B310
        target.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            observed_size += len(chunk)
            if observed_size > expected_size or observed_size > MAX_ARTIFACT_BYTES:
                raise RuntimeError("PyPI artifact exceeds its declared safe size")
            digest.update(chunk)
            output.write(chunk)
    if observed_size != expected_size:
        raise RuntimeError(
            f"PyPI artifact size mismatch: expected {expected_size}, "
            f"observed {observed_size}"
        )
    return digest.hexdigest()


def verify_release(version: str, *, attempts: int, delay: float) -> dict[str, Any]:
    """Wait for and verify exactly one public wheel and source distribution."""
    payload: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            candidate = _get_json(PYPI_JSON)
        except urllib.error.URLError:
            candidate = {}
        if version in candidate.get("releases", {}):
            payload = candidate
            break
        if attempt < attempts:
            time.sleep(delay)
    if payload is None:
        raise RuntimeError(f"PyPI version {version} did not become visible")
    if payload.get("info", {}).get("version") != version:
        raise RuntimeError(f"PyPI latest version is not {version}")

    files = payload["releases"][version]
    if len(files) != 2 or {item["packagetype"] for item in files} != {
        "bdist_wheel",
        "sdist",
    }:
        raise RuntimeError("PyPI must expose exactly one wheel and one sdist")
    if any(item.get("yanked") for item in files):
        raise RuntimeError("PyPI release contains a yanked artifact")

    artifact_receipts = []
    with tempfile.TemporaryDirectory(prefix="mettle-pypi-") as temporary:
        directory = Path(temporary)
        wheel: Path | None = None
        for item in sorted(files, key=lambda entry: entry["filename"]):
            filename = item["filename"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise RuntimeError(f"unsafe PyPI artifact filename: {filename!r}")
            target = directory / filename
            observed = _download(item["url"], target, int(item["size"]))
            expected = item["digests"]["sha256"]
            if observed != expected:
                raise RuntimeError(f"PyPI digest mismatch for {target.name}")
            if item["packagetype"] == "bdist_wheel":
                wheel = target
            artifact_receipts.append(
                {
                    "filename": item["filename"],
                    "packagetype": item["packagetype"],
                    "sha256": observed,
                    "size": item["size"],
                    "upload_time_iso_8601": item["upload_time_iso_8601"],
                    "url": item["url"],
                }
            )
        if wheel is None:
            raise RuntimeError("PyPI wheel is missing")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            server_path = "mettle/mcp_server.py"
            if server_path not in names:
                raise RuntimeError("public wheel does not contain the MCP server")
            server = archive.read(server_path).decode("utf-8")
            if "mettle_auto_verify" in server:
                raise RuntimeError(
                    "public wheel contains the forbidden automatic solver"
                )
            entry_points = [
                name for name in names if name.endswith("/entry_points.txt")
            ]
            if len(entry_points) != 1:
                raise RuntimeError("public wheel entry points are missing or ambiguous")
            entry_point_text = archive.read(entry_points[0]).decode("utf-8")
            if "mettle-mcp = mettle.mcp_server:main" not in entry_point_text:
                raise RuntimeError(
                    "public wheel does not expose the expected mettle-mcp"
                )
            metadata_paths = [name for name in names if name.endswith("/METADATA")]
            if len(metadata_paths) != 1:
                raise RuntimeError("public wheel metadata is missing or ambiguous")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if (
                metadata.get("Name") != "mettle-verifier"
                or metadata.get("Version") != version
            ):
                raise RuntimeError(
                    "public wheel identity differs from the requested release"
                )

    return {
        "schema_version": "1.0",
        "checked_at": datetime.now(UTC).isoformat(),
        "project": "mettle-verifier",
        "version": version,
        "public_index": PYPI_JSON,
        "automatic_solver_absent": True,
        "artifacts": artifact_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=18)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_release(args.version, attempts=args.attempts, delay=args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
