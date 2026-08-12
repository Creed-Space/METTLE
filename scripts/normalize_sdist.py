"""Normalize a Python source distribution into a deterministic tarball."""

from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(ValueError):
    """Raised when an sdist contains a member unsafe to reproduce."""


@dataclass(frozen=True)
class _Member:
    name: str
    is_directory: bool
    executable: bool
    data: bytes


def _validate_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise UnsafeArchiveError(f"unsafe archive member: {name!r}")


def _read_members(archive: Path) -> list[_Member]:
    members: list[_Member] = []
    names: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as source:
        for item in source.getmembers():
            _validate_name(item.name)
            if item.name in names:
                raise UnsafeArchiveError(f"duplicate archive member: {item.name!r}")
            names.add(item.name)
            if item.isdir():
                data = b""
            elif item.isfile():
                extracted = source.extractfile(item)
                if extracted is None:
                    raise UnsafeArchiveError(
                        f"could not read regular archive member: {item.name!r}"
                    )
                data = extracted.read()
            else:
                raise UnsafeArchiveError(
                    f"unsupported archive member type for {item.name!r}"
                )
            members.append(
                _Member(
                    name=item.name,
                    is_directory=item.isdir(),
                    executable=bool(item.mode & 0o111),
                    data=data,
                )
            )
    return sorted(members, key=lambda member: member.name)


def normalize_sdist(archive: Path, source_date_epoch: int) -> None:
    """Rewrite ``archive`` with stable ordering, ownership, modes, and times."""
    if source_date_epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")
    members = _read_members(archive)
    if not members:
        raise UnsafeArchiveError("source distribution is empty")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=source_date_epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as target:
                    for member in members:
                        item = tarfile.TarInfo(member.name)
                        item.type = (
                            tarfile.DIRTYPE if member.is_directory else tarfile.REGTYPE
                        )
                        item.mode = (
                            0o755 if member.is_directory or member.executable else 0o644
                        )
                        item.uid = 0
                        item.gid = 0
                        item.uname = ""
                        item.gname = ""
                        item.mtime = source_date_epoch
                        item.size = len(member.data)
                        if member.is_directory:
                            target.addfile(item)
                        else:
                            target.addfile(item, BytesIO(member.data))
        temporary.chmod(0o644)
        os.replace(temporary, archive)
        _read_members(archive)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    normalize_sdist(args.archive, args.source_date_epoch)


if __name__ == "__main__":
    main()
