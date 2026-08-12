"""Build METTLE distributions with source-bound reproducible metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil

# Release tooling uses fixed argv arrays and never invokes a shell.
import subprocess  # nosec B404
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scripts.normalize_sdist import normalize_sdist


ROOT = Path(__file__).resolve().parents[1]


def _git_value(format_string: str, revision: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to bind distributions to source")
    # The executable is resolved absolutely and every argument is passed verbatim.
    return subprocess.check_output(  # nosec
        [git, "show", "-s", f"--format={format_string}", revision],
        cwd=ROOT,
        text=True,
    ).strip()


def _require_exact_source(source_sha: str, *, allow_dirty: bool) -> None:
    """Refuse to label artifacts with a revision other than checkout HEAD."""
    head_sha = _git_value("%H", "HEAD")
    if source_sha != head_sha:
        raise RuntimeError(
            f"source revision resolves to {source_sha}, but checkout HEAD is {head_sha}"
        )
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to bind distributions to source")
    if not allow_dirty:
        status = subprocess.check_output(  # nosec
            [git, "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
        ).strip()
        if status:
            changed = [line[3:] for line in status.splitlines()[:5]]
            suffix = "" if len(status.splitlines()) <= 5 else ", ..."
            raise RuntimeError(
                "distribution source checkout is not clean: "
                f"{', '.join(changed)}{suffix}"
            )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def build_distributions(
    *,
    outdir: Path,
    source_revision: str,
    source_date_epoch: int | None = None,
    builder_id: str = "local",
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Build one wheel and sdist, then return a secret-free build receipt."""
    source_sha = _git_value("%H", source_revision)
    _require_exact_source(source_sha, allow_dirty=allow_dirty)
    epoch = (
        source_date_epoch
        if source_date_epoch is not None
        else int(_git_value("%ct", source_revision))
    )
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")

    resolved_outdir = outdir.resolve()
    resolved_outdir.mkdir(parents=True, exist_ok=True)
    if any(resolved_outdir.iterdir()):
        raise RuntimeError(
            f"distribution output directory is not empty: {resolved_outdir}"
        )
    environment = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch)}
    # The executable is this trusted interpreter and the module argv is fixed.
    subprocess.run(  # nosec B603
        [sys.executable, "-m", "build", "--outdir", str(resolved_outdir)],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    sdists = sorted(resolved_outdir.glob("*.tar.gz"))
    wheels = sorted(resolved_outdir.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise RuntimeError(
            "expected exactly one wheel and one source distribution, found "
            f"{len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    normalize_sdist(sdists[0], epoch)

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    artifacts = [
        {
            "name": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size": artifact.stat().st_size,
        }
        for artifact in sorted([*wheels, *sdists], key=lambda path: path.name)
    ]
    return {
        "schema_version": "1.0",
        "builder_id": builder_id,
        "source_sha": source_sha,
        "source_date_epoch": epoch,
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "toolchain": {
            "build_frontend": _package_version("build"),
            "build_backend_requires": pyproject["build-system"]["requires"],
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "dist")
    parser.add_argument("--source-revision", default="HEAD")
    parser.add_argument("--source-date-epoch", type=int)
    parser.add_argument("--builder-id", default="local")
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit local candidate builds before commit; release automation omits this",
    )
    args = parser.parse_args()
    receipt = build_distributions(
        outdir=args.outdir,
        source_revision=args.source_revision,
        source_date_epoch=args.source_date_epoch,
        builder_id=args.builder_id,
        allow_dirty=args.allow_dirty,
    )
    output = args.metadata_output or args.outdir / "BUILD-ENVIRONMENT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
