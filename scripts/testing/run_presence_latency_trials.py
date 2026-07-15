#!/usr/bin/env python3
"""Calibrate live METTLE Presence timing with direct and paced cohorts.

The paced cohorts simulate delay. They do not represent measured humans and
must not be used alone to set or enforce a product threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.testing.presence_trial_support import (
    BRONZE_SUITES,
    PresenceSessionDriver,
    TrialFailure,
    presence_timing_receipts,
)


@dataclass(frozen=True)
class Cohort:
    name: str
    delay_ms: int
    sessions: int


def parse_cohort(value: str) -> Cohort:
    """Parse NAME:DELAY_MS:SESSIONS."""
    try:
        name, delay_text, sessions_text = value.split(":", 2)
        delay_ms = int(delay_text)
        sessions = int(sessions_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "cohort must use NAME:DELAY_MS:SESSIONS"
        ) from exc
    if not name or delay_ms < 0 or sessions < 1:
        raise argparse.ArgumentTypeError(
            "cohort name must be non-empty, delay non-negative, and sessions positive"
        )
    return Cohort(name=name, delay_ms=delay_ms, sessions=sessions)


def percentile(values: list[int], proportion: float) -> int:
    """Return a nearest-rank percentile for a non-empty integer sample."""
    if not values:
        raise ValueError("Percentile sample cannot be empty")
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * proportion) + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Timing sample cannot be empty")
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def _run_cohort(
    *, base_url: str, api_key: str, cohort: Cohort, timeout: float
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for session_index in range(1, cohort.sessions + 1):
        with PresenceSessionDriver(
            base_url=base_url,
            api_key=api_key,
            suites=list(BRONZE_SUITES),
            audience="latency-calibration.mettle.local",
            timeout_seconds=timeout,
        ) as driver:
            try:
                result = driver.complete(delay_ms=cohort.delay_ms)
            except TrialFailure as exc:
                if "returned 429" not in str(exc):
                    raise
                time.sleep(61)
                result = driver.complete(delay_ms=cohort.delay_ms)
            receipts = presence_timing_receipts(result)
            if len(receipts) != len(driver.observations):
                raise TrialFailure(
                    "Client and signed server timing receipt counts differ"
                )
            for observation, receipt in zip(driver.observations, receipts, strict=True):
                if observation.action != receipt.get("action"):
                    raise TrialFailure("Client and signed server timing actions differ")
                samples.append(
                    {
                        "cohort": cohort.name,
                        "session_index": session_index,
                        "action": observation.action,
                        "configured_delay_ms": observation.configured_delay_ms,
                        "client_solve_time_ms": observation.solve_time_ms,
                        "client_request_time_ms": observation.request_time_ms,
                        "server_response_time_ms": receipt["response_time_ms"],
                    }
                )
    return samples


def build_report(
    *, base_url: str, cohorts: list[Cohort], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for cohort in cohorts:
        values = [
            int(sample["server_response_time_ms"])
            for sample in samples
            if sample["cohort"] == cohort.name
        ]
        summaries[cohort.name] = {
            "configured_delay_ms": cohort.delay_ms,
            **summarize(values),
        }

    direct = next((cohort for cohort in cohorts if cohort.delay_ms == 0), None)
    separation: dict[str, Any] = {}
    if direct is not None:
        direct_values = [
            int(sample["server_response_time_ms"])
            for sample in samples
            if sample["cohort"] == direct.name
        ]
        for cohort in cohorts:
            if cohort is direct:
                continue
            paced_values = [
                int(sample["server_response_time_ms"])
                for sample in samples
                if sample["cohort"] == cohort.name
            ]
            gap = min(paced_values) - max(direct_values)
            separation[cohort.name] = {
                "direct_max_to_paced_min_gap_ms": gap,
                "observed_overlap": gap <= 0,
                "descriptive_midpoint_ms": (
                    (max(direct_values) + min(paced_values)) // 2 if gap > 0 else None
                ),
            }

    return {
        "schema": "mettle-presence-latency-calibration-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": base_url,
        "suites": list(BRONZE_SUITES),
        "attestation_signature_verified": True,
        "cohorts": [asdict(cohort) for cohort in cohorts],
        "summaries": summaries,
        "separation": separation,
        "interpretation_limits": [
            "Paced cohorts are synthetic delay injections, not measured humans.",
            "Network location, load, solver architecture, and warmup affect timing.",
            "A cooperating solver with the holder key can answer without human relay.",
            "A product threshold must not be enforced from this artifact alone.",
        ],
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--api-key-env",
        default="METTLE_API_KEY",
        help="Environment variable containing the bearer key",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        type=parse_cohort,
        dest="cohorts",
        help="Repeatable NAME:DELAY_MS:SESSIONS specification",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = args.cohorts or [
        Cohort("direct", 0, 3),
        Cohort("paced_250", 250, 3),
        Cohort("paced_1000", 1000, 3),
    ]
    if len({cohort.name for cohort in cohorts}) != len(cohorts):
        parser.error("cohort names must be unique")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"{args.api_key_env} is unset or empty")

    samples: list[dict[str, Any]] = []
    for cohort in cohorts:
        samples.extend(
            _run_cohort(
                base_url=args.base_url,
                api_key=api_key,
                cohort=cohort,
                timeout=args.timeout,
            )
        )
    report = build_report(
        base_url=args.base_url.rstrip("/"), cohorts=cohorts, samples=samples
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summaries": report["summaries"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
