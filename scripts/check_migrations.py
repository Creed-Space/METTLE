#!/usr/bin/env python3
"""Report or apply METTLE's forward-only database schema migrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database


def build_report(*, apply: bool = False) -> dict[str, object]:
    """Return a non-sensitive migration report, optionally applying upgrades."""
    if apply:
        database.init_db()
    current = database.get_schema_version()
    latest = database.LATEST_SCHEMA_VERSION
    scheme = urlparse(database.DATABASE_URL).scheme or "unknown"
    healthy = database.check_health()
    return {
        "schema": "mettle-migration-status-v1",
        "database_scheme": scheme,
        "database_healthy": healthy,
        "current_version": current,
        "latest_version": latest,
        "current": healthy and current == latest,
        "action": "apply" if apply else "check",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="inspect without mutation")
    mode.add_argument(
        "--apply",
        action="store_true",
        help="apply pending forward-only migrations, then verify",
    )
    args = parser.parse_args(argv)
    report = build_report(apply=args.apply)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["current"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
