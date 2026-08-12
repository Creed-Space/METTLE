"""Compare distribution artifacts produced by independent builders."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class ReproducibilityError(ValueError):
    """Raised when independent distribution builds do not agree."""


def _files(directory: Path) -> dict[str, Path]:
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.name.endswith((".whl", ".tar.gz"))
    }
    if (
        len([name for name in files if name.endswith(".whl")]) != 1
        or len([name for name in files if name.endswith(".tar.gz")]) != 1
    ):
        raise ReproducibilityError(
            f"{directory} must contain exactly one wheel and one sdist"
        )
    return files


def _validate_builder_receipt(
    label: str, receipt: dict[str, Any], artifacts: dict[str, Path]
) -> None:
    if receipt.get("schema_version") != "1.0":
        raise ReproducibilityError(f"unsupported builder receipt schema for {label}")
    if receipt.get("builder_id") != label:
        raise ReproducibilityError(
            f"builder receipt identity differs for {label}: "
            f"{receipt.get('builder_id')!r}"
        )
    declared = receipt.get("artifacts")
    if not isinstance(declared, list):
        raise ReproducibilityError(f"builder receipt artifacts are invalid for {label}")
    declared_by_name = {item.get("name"): item for item in declared}
    if len(declared_by_name) != len(declared) or set(declared_by_name) != set(
        artifacts
    ):
        raise ReproducibilityError(
            f"builder receipt artifact names differ from files for {label}"
        )
    for name, path in artifacts.items():
        item = declared_by_name[name]
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            item.get("sha256") != observed_hash
            or item.get("size") != path.stat().st_size
        ):
            raise ReproducibilityError(
                f"builder receipt does not bind artifact {name} for {label}"
            )


def compare_builds(
    builders: dict[str, Path], *, min_linux_builders: int, require_macos: bool
) -> dict[str, Any]:
    """Require byte identity plus source and toolchain agreement."""
    if len(builders) < 2:
        raise ReproducibilityError("at least two independent builders are required")
    receipts: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Path]] = {}
    for label, directory in sorted(builders.items()):
        receipt_path = directory / "BUILD-ENVIRONMENT.json"
        receipts[label] = json.loads(receipt_path.read_text(encoding="utf-8"))
        files[label] = _files(directory)
        _validate_builder_receipt(label, receipts[label], files[label])

    first_label = sorted(builders)[0]
    names = set(files[first_label])
    for label, artifacts in files.items():
        if set(artifacts) != names:
            raise ReproducibilityError(
                f"artifact names differ for {label}: {sorted(artifacts)}"
            )

    common_fields = ("source_sha", "source_date_epoch", "toolchain")
    for field in common_fields:
        values = {
            json.dumps(receipt[field], sort_keys=True) for receipt in receipts.values()
        }
        if len(values) != 1:
            raise ReproducibilityError(f"builders disagree on {field}")
    for field in ("python_implementation", "python_version"):
        values = {receipt["environment"].get(field) for receipt in receipts.values()}
        if len(values) != 1:
            raise ReproducibilityError(f"builders disagree on environment.{field}")

    platforms = Counter(
        str(receipt["environment"]["platform"]) for receipt in receipts.values()
    )
    if platforms["Linux"] < min_linux_builders:
        raise ReproducibilityError(
            f"need {min_linux_builders} Linux builders, observed {platforms['Linux']}"
        )
    if require_macos and platforms["Darwin"] < 1:
        raise ReproducibilityError("a macOS builder is required")

    artifact_receipts = []
    for name in sorted(names):
        digests = {
            label: hashlib.sha256(artifacts[name].read_bytes()).hexdigest()
            for label, artifacts in files.items()
        }
        if len(set(digests.values())) != 1:
            raise ReproducibilityError(
                f"artifact {name} differs: {json.dumps(digests, sort_keys=True)}"
            )
        artifact_receipts.append(
            {
                "name": name,
                "sha256": next(iter(digests.values())),
                "builders": sorted(digests),
            }
        )

    return {
        "schema_version": "1.0",
        "result": "byte-identical",
        "source_sha": receipts[first_label]["source_sha"],
        "source_date_epoch": receipts[first_label]["source_date_epoch"],
        "toolchain": receipts[first_label]["toolchain"],
        "platform_counts": dict(sorted(platforms.items())),
        "builders": [receipts[label] for label in sorted(receipts)],
        "artifacts": artifact_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builder",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Independent builder label and downloaded artifact directory",
    )
    parser.add_argument("--min-linux-builders", type=int, default=2)
    parser.add_argument("--require-macos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    builders: dict[str, Path] = {}
    for value in args.builder:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in builders:
            parser.error(f"invalid or duplicate --builder value: {value!r}")
        builders[label] = Path(path)
    receipt = compare_builds(
        builders,
        min_linux_builders=args.min_linux_builders,
        require_macos=args.require_macos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
