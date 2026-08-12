#!/usr/bin/env python3
"""Write and verify content fingerprints on long-lived static assets."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
HASH_LENGTH = 12
SHORT_LIVED = {"site.webmanifest", "robots.txt", "sitemap.xml"}
REFERENCE = re.compile(
    r"(?P<url>(?:https://mettle\.sh)?/static/"
    r"(?P<path>[A-Za-z0-9_./-]+))"
    r"(?:\?v=(?P<version>[0-9a-f]+))?"
)


def asset_version(path: Path) -> str:
    """Return the stable public version for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_LENGTH]


def is_fingerprintable(relative_path: str) -> bool:
    """Return whether a static resource belongs in immutable caching."""
    return Path(relative_path).name not in SHORT_LIVED


def _rewrite_source(source: Path) -> str:
    text = source.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        relative = match.group("path")
        if not is_fingerprintable(relative):
            return match.group("url")
        target = STATIC / relative
        if not target.is_file():
            raise FileNotFoundError(f"{source.relative_to(ROOT)} references {target}")
        return f"{match.group('url')}?v={asset_version(target)}"

    return REFERENCE.sub(replace, text)


def update(*, check: bool) -> list[Path]:
    """Update sources, or return stale sources without writing in check mode."""
    stale: list[Path] = []
    # CSS first because HTML fingerprints the final CSS bytes.
    sources = [*sorted(STATIC.rglob("*.css")), *sorted(STATIC.glob("*.html"))]
    for source in sources:
        current = source.read_text(encoding="utf-8")
        expected = _rewrite_source(source)
        if current == expected:
            continue
        stale.append(source)
        if not check:
            source.write_text(expected, encoding="utf-8")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any checked-in fingerprint is stale",
    )
    args = parser.parse_args()
    stale = update(check=args.check)
    if stale and args.check:
        for path in stale:
            print(f"stale fingerprint: {path.relative_to(ROOT)}")
        return 1
    for path in stale:
        print(f"updated: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
