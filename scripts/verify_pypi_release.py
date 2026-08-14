"""Verify one public PyPI release and emit a hash-bound receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path
from typing import Any, cast


PYPI_PROJECT = "https://pypi.org/pypi/mettle-verifier"
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
MCP_SERVER_NAME = "io.github.Creed-Space/mettle-mcp"


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Reject redirects before urllib can forward a request to another origin."""

    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        _msg: str,
        headers: object,
        _newurl: str,
    ) -> None:
        raise urllib.error.HTTPError(
            getattr(req, "full_url", "https://pypi.org"),
            code,
            "redirect rejected",
            cast(Any, headers),
            cast(Any, fp),
        )


_HTTPS_OPENER = urllib.request.build_opener(_RejectRedirects())


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
    with _HTTPS_OPENER.open(request, timeout=30) as response:  # nosec B310
        _require_https(response.geturl(), {"pypi.org"})
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("PyPI returned a non-object response")
    return payload


def _version_json_url(version: str) -> str:
    """Build the exact-version PyPI endpoint without permitting path injection."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}", version):
        raise ValueError(f"unsafe PyPI version: {version!r}")
    return f"{PYPI_PROJECT}/{urllib.parse.quote(version, safe='')}/json"


def _require_mcp_ownership_marker(description: str) -> str:
    """Require the package proof consumed by Official MCP Registry validation."""
    marker = f"mcp-name: {MCP_SERVER_NAME}"
    if marker not in description:
        raise RuntimeError("PyPI description is missing the MCP ownership marker")
    return marker


def _load_reproducibility_receipt(
    path: Path,
    *,
    version: str,
    source_sha: str,
) -> dict[str, str]:
    """Load the independent source-bound artifact hashes before public execution."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA must be a lowercase 40-character Git SHA")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0"
        or payload.get("result") != "byte-identical"
        or payload.get("source_sha") != source_sha
    ):
        raise ValueError("reproducibility receipt is not bound to this source")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("reproducibility receipt must declare exactly two artifacts")
    expected: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("reproducibility receipt contains an invalid artifact")
        name = artifact.get("name")
        digest = artifact.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or name in expected
        ):
            raise ValueError("reproducibility receipt contains an invalid artifact")
        expected[name] = digest
    wheels = [
        name
        for name in expected
        if name.startswith(f"mettle_verifier-{version}-") and name.endswith(".whl")
    ]
    sdists = [name for name in expected if name == f"mettle_verifier-{version}.tar.gz"]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            "reproducibility artifact names differ from the release version"
        )
    return expected


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
        _HTTPS_OPENER.open(request, timeout=60) as response,  # nosec B310
        target.open("wb") as output,
    ):
        _require_https(response.geturl(), {"files.pythonhosted.org"})
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


def verify_release(
    version: str,
    *,
    attempts: int,
    delay: float,
    reproducibility_path: Path,
    source_sha: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Wait for and verify exactly one public wheel and source distribution."""
    reproducible_hashes = _load_reproducibility_receipt(
        reproducibility_path,
        version=version,
        source_sha=source_sha,
    )
    public_index = _version_json_url(version)
    payload: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            candidate = _get_json(public_index)
        except urllib.error.URLError:
            candidate = {}
        if candidate.get("info", {}).get("version") == version:
            payload = candidate
            break
        if attempt < attempts:
            time.sleep(delay)
    if payload is None:
        raise RuntimeError(f"PyPI version {version} did not become visible")
    ownership_marker = _require_mcp_ownership_marker(
        str(payload.get("info", {}).get("description", ""))
    )

    files = payload.get("urls", [])
    if len(files) != 2 or {item["packagetype"] for item in files} != {
        "bdist_wheel",
        "sdist",
    }:
        raise RuntimeError("PyPI must expose exactly one wheel and one sdist")
    if any(item.get("yanked") for item in files):
        raise RuntimeError("PyPI release contains a yanked artifact")
    if {item.get("filename") for item in files} != set(reproducible_hashes):
        raise RuntimeError("PyPI artifact names differ from independent builds")

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
            if observed != reproducible_hashes[filename]:
                raise RuntimeError(
                    f"PyPI artifact differs from independent build: {target.name}"
                )
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
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            targets = {name: artifact_dir / name for name in reproducible_hashes}
            if any(target.exists() for target in targets.values()):
                raise RuntimeError("public artifact destination is not empty")
            for name, target in targets.items():
                with (
                    (directory / name).open("rb") as source,
                    target.open("xb") as output,
                ):
                    shutil.copyfileobj(source, output)

    return {
        "schema_version": "1.0",
        "checked_at": datetime.now(UTC).isoformat(),
        "project": "mettle-verifier",
        "version": version,
        "public_index": public_index,
        "mcp_ownership_marker": ownership_marker,
        "automatic_solver_absent": True,
        "artifacts": artifact_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=18)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_release(
        args.version,
        attempts=args.attempts,
        delay=args.delay,
        reproducibility_path=args.reproducibility,
        source_sha=args.source_sha,
        artifact_dir=args.artifact_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
