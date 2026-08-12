"""Regression tests for deterministic Python distribution tooling."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.compare_distributions import ReproducibilityError, compare_builds
from scripts.build_distributions import _require_exact_source
from scripts.normalize_sdist import UnsafeArchiveError, normalize_sdist


def _write_archive(path: Path, *, mtime: int, unsafe: bool = False) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename=path.name, mtime=mtime
        ) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                directory = tarfile.TarInfo("package-1.0")
                directory.type = tarfile.DIRTYPE
                directory.mode = 0o775
                directory.uid = 501
                directory.gid = 20
                directory.uname = "builder"
                directory.gname = "staff"
                directory.mtime = mtime + 0.25
                tar.addfile(directory)
                item = tarfile.TarInfo(
                    "../escape" if unsafe else "package-1.0/data.txt"
                )
                item.size = 7
                item.mode = 0o664
                item.uid = 501
                item.gid = 20
                item.uname = "builder"
                item.gname = "staff"
                item.mtime = mtime + 0.5
                tar.addfile(item, io.BytesIO(b"payload"))


def test_normalized_sdists_are_byte_identical_and_portable(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_archive(first, mtime=1_700_000_000)
    _write_archive(second, mtime=1_800_000_000)

    epoch = 1_750_000_000
    normalize_sdist(first, epoch)
    normalize_sdist(second, epoch)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "package-1.0",
            "package-1.0/data.txt",
        ]
        assert all(member.mtime == epoch for member in members)
        assert all(member.uid == member.gid == 0 for member in members)
        assert all(member.uname == member.gname == "" for member in members)
        assert members[0].mode == 0o755
        assert members[1].mode == 0o644
        assert archive.extractfile(members[1]).read() == b"payload"  # type: ignore[union-attr]


def test_normalizer_rejects_unsafe_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_archive(archive, mtime=1_700_000_000, unsafe=True)
    with pytest.raises(UnsafeArchiveError, match="unsafe archive member"):
        normalize_sdist(archive, 1_750_000_000)


def test_distribution_source_requires_checkout_head_and_clean_tree() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    _require_exact_source(head, allow_dirty=True)
    with pytest.raises(RuntimeError, match="checkout HEAD"):
        _require_exact_source("0" * 40, allow_dirty=True)
    with pytest.raises(RuntimeError, match="checkout is not clean"):
        _require_exact_source(head, allow_dirty=False)


def test_distribution_builder_refuses_nonempty_output(tmp_path: Path) -> None:
    from scripts.build_distributions import build_distributions

    outdir = tmp_path / "dist"
    outdir.mkdir()
    (outdir / "unexpected").write_text("preserve me")
    with pytest.raises(RuntimeError, match="output directory is not empty"):
        build_distributions(
            outdir=outdir,
            source_revision="HEAD",
            builder_id="test",
            allow_dirty=True,
        )
    assert (outdir / "unexpected").read_text() == "preserve me"


def _builder(
    root: Path, label: str, platform_name: str, *, wheel: bytes = b"wheel"
) -> Path:
    directory = root / label
    directory.mkdir()
    (directory / "package-1.0-py3-none-any.whl").write_bytes(wheel)
    (directory / "package-1.0.tar.gz").write_bytes(b"sdist")
    artifacts = []
    for path in sorted(directory.iterdir()):
        artifacts.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": "1.0",
        "builder_id": label,
        "source_sha": "a" * 40,
        "source_date_epoch": 1_750_000_000,
        "environment": {
            "platform": platform_name,
            "python_implementation": "CPython",
            "python_version": "3.11.9",
        },
        "toolchain": {"build_frontend": "1.5.0", "build_backend_requires": []},
        "artifacts": artifacts,
    }
    (directory / "BUILD-ENVIRONMENT.json").write_text(json.dumps(receipt))
    return directory


def test_compare_builds_requires_two_linux_and_one_macos(tmp_path: Path) -> None:
    report = compare_builds(
        {
            "linux-1": _builder(tmp_path, "linux-1", "Linux"),
            "linux-2": _builder(tmp_path, "linux-2", "Linux"),
            "macos": _builder(tmp_path, "macos", "Darwin"),
        },
        min_linux_builders=2,
        require_macos=True,
    )
    assert report["result"] == "byte-identical"
    assert report["platform_counts"] == {"Darwin": 1, "Linux": 2}


def test_compare_builds_rejects_byte_drift(tmp_path: Path) -> None:
    with pytest.raises(ReproducibilityError, match="differs"):
        compare_builds(
            {
                "one": _builder(tmp_path, "one", "Linux"),
                "two": _builder(tmp_path, "two", "Linux", wheel=b"different"),
            },
            min_linux_builders=2,
            require_macos=False,
        )


def test_compare_builds_rejects_tampered_builder_receipt(tmp_path: Path) -> None:
    one = _builder(tmp_path, "one", "Linux")
    two = _builder(tmp_path, "two", "Linux")
    receipt_path = two / "BUILD-ENVIRONMENT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(ReproducibilityError, match="does not bind artifact"):
        compare_builds(
            {"one": one, "two": two},
            min_linux_builders=2,
            require_macos=False,
        )
