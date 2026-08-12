#!/usr/bin/env python3
"""Measure exact replay value in the public quick-challenge generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mettle.challenger import generate_challenge_set  # noqa: E402
from mettle.models import Difficulty  # noqa: E402
from mettle.protocol import SUITE_POLICY_VERSION  # noqa: E402


def public_shape(challenge) -> str:
    sanitized = challenge.sanitized().model_dump(mode="json")
    for volatile in ("id", "issued_at", "expires_at"):
        sanitized.pop(volatile, None)
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"))


def evaluate(samples: int) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for _ in range(samples):
        for challenge in generate_challenge_set(Difficulty.BASIC):
            by_type[challenge.type.value].append(public_shape(challenge))

    metrics = []
    for challenge_type, values in sorted(by_type.items()):
        counts = Counter(values)
        midpoint = len(values) // 2
        learned = set(values[:midpoint])
        later = values[midpoint:]
        max_frequency = max(counts.values())
        metrics.append(
            {
                "challenge_type": challenge_type,
                "samples": len(values),
                "unique_public_shapes": len(counts),
                "collision_rate": 1 - (len(counts) / len(values)),
                "empirical_min_entropy_bits_lower_bound": round(
                    -math.log2(max_frequency / len(values)), 4
                ),
                "adaptive_replay_coverage": (
                    sum(value in learned for value in later) / len(later)
                    if later
                    else 0.0
                ),
                "corpus_digest": hashlib.sha256(
                    "\n".join(sorted(counts)).encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "report_schema_version": "1.0",
        "suite_policy_version": SUITE_POLICY_VERSION,
        "scope": "public quick API basic challenge generator",
        "samples_per_challenge_type": samples,
        "metrics": metrics,
        "rotation_triggers": {
            "collision_rate_max": 0.01,
            "adaptive_replay_coverage_max": 0.05,
            "response": (
                "Treat an exceeded trigger as evidence of a finite harvestable corpus. "
                "Version and replace the affected generator before strengthening claims."
            ),
        },
        "limitations": [
            "Observed uniqueness is a lower bound, not a proof of generator entropy.",
            "This run does not model semantic answer transfer or the authenticated suites.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 100 <= args.samples <= 100000:
        parser.error("--samples must be between 100 and 100000")
    rendered = json.dumps(evaluate(args.samples), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
