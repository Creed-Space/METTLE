#!/usr/bin/env python3
"""Update METTLE's bounded copyright year in the four public HTML pages."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
START_YEAR = 2026
PAGES = ("index.html", "docs.html", "test.html", "about.html")
COPYRIGHT = re.compile(r"&copy; 2026(?:-[0-9]{2})? Creed Space\.")


def copyright_label(year: int) -> str:
    """Return the canonical visible copyright prefix for a target year."""
    if year < START_YEAR:
        raise ValueError(f"year must be {START_YEAR} or later")
    if year == START_YEAR:
        return "&copy; 2026 Creed Space."
    return f"&copy; 2026-{year % 100:02d} Creed Space."


def update_file(path: Path, year: int, *, check: bool) -> bool:
    """Update exactly one canonical footer; return whether it was stale."""
    current = path.read_text(encoding="utf-8")
    matches = COPYRIGHT.findall(current)
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected one canonical copyright, found {len(matches)}"
        )
    expected = COPYRIGHT.sub(copyright_label(year), current)
    stale = current != expected
    if stale and not check:
        path.write_text(expected, encoding="utf-8")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="UTC calendar year to apply",
    )
    parser.add_argument("--check", action="store_true", help="check without writing")
    args = parser.parse_args()

    stale: list[Path] = []
    for name in PAGES:
        path = STATIC / name
        if update_file(path, args.year, check=args.check):
            stale.append(path)
    if stale and args.check:
        for path in stale:
            print(f"stale footer: {path.relative_to(ROOT)}")
        return 1
    for path in stale:
        print(f"updated: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
