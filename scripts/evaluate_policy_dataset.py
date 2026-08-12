#!/usr/bin/env python3
"""Aggregate a privacy-minimal held-out METTLE decision dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ALLOWED_FIELDS = {
    "dataset_version",
    "subject_class",
    "suite",
    "expected_pass",
    "observed_pass",
    "cohort",
}
SUBJECT_CLASSES = {"becoming-mind", "human-assisted"}
MIN_POSITIVE = 30
MIN_NEGATIVE = 30


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - ALLOWED_FIELDS:
            raise ValueError(f"line {line_number}: unknown or privacy-sensitive fields")
        required = ALLOWED_FIELDS - {"cohort"}
        if not required <= set(row):
            raise ValueError(f"line {line_number}: missing required fields")
        if (
            not isinstance(row["dataset_version"], str)
            or not 1 <= len(row["dataset_version"]) <= 64
        ):
            raise ValueError(f"line {line_number}: invalid dataset_version")
        if row["subject_class"] not in SUBJECT_CLASSES:
            raise ValueError(f"line {line_number}: invalid subject_class")
        if not isinstance(row["suite"], str) or not 1 <= len(row["suite"]) <= 128:
            raise ValueError(f"line {line_number}: invalid suite")
        if not isinstance(row["expected_pass"], bool) or not isinstance(
            row["observed_pass"], bool
        ):
            raise ValueError(f"line {line_number}: decisions must be Boolean")
        if "cohort" in row and (
            not isinstance(row["cohort"], str) or not 1 <= len(row["cohort"]) <= 64
        ):
            raise ValueError(f"line {line_number}: invalid cohort")
        rows.append(row)
    if not rows:
        raise ValueError("evaluation dataset is empty")
    if len({row["dataset_version"] for row in rows}) != 1:
        raise ValueError("all rows must use one dataset_version")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    )
    for row in rows:
        cell = cells[(row["suite"], row["subject_class"])]
        expected = row["expected_pass"]
        observed = row["observed_pass"]
        cell[
            "tp"
            if expected and observed
            else "fn"
            if expected
            else "fp"
            if observed
            else "tn"
        ] += 1
    results = []
    for (suite, subject_class), counts in sorted(cells.items()):
        positives = counts["tp"] + counts["fn"]
        negatives = counts["tn"] + counts["fp"]
        results.append(
            {
                "suite": suite,
                "subject_class": subject_class,
                "counts": counts,
                "false_reject_rate": counts["fn"] / positives if positives else None,
                "false_accept_rate": counts["fp"] / negatives if negatives else None,
                "insufficient_data": positives < MIN_POSITIVE
                or negatives < MIN_NEGATIVE,
            }
        )
    return {
        "aggregate_schema_version": "1.0",
        "dataset_version": rows[0]["dataset_version"],
        "records": len(rows),
        "minimum_examples_per_decision_class": MIN_POSITIVE,
        "results": results,
        "decision_authority": "human protocol governance review required",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(load_rows(args.dataset))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
