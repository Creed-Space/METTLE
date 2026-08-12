#!/usr/bin/env python3
"""Run a bounded local load characterization with explicit regression budgets."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import redis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from mettle.challenger import (  # noqa: E402
    generate_challenge_set,
    generate_speed_math_challenge,
)
from mettle.models import Difficulty  # noqa: E402
from mettle.verifier import verify_response  # noqa: E402

BUDGET_PATH = ROOT / "benchmarks" / "bounded-load-budget.json"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile) - 1))
    return ordered[index]


def measure(
    name: str, operation: Callable[[int], None], *, operations: int, concurrency: int
) -> dict:
    durations: list[float] = []
    errors: list[str] = []

    def timed(index: int) -> float:
        started = time.perf_counter()
        operation(index)
        return (time.perf_counter() - started) * 1000

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(timed, index) for index in range(operations)]
        for future in as_completed(futures):
            try:
                durations.append(future.result())
            except Exception as exc:
                errors.append(type(exc).__name__)
    wall_seconds = time.perf_counter() - wall_started
    if not durations:
        raise RuntimeError(f"{name} produced no successful measurements")
    return {
        "name": name,
        "operations": operations,
        "concurrency": concurrency,
        "throughput_per_second": round(operations / wall_seconds, 2),
        "p50_ms": round(statistics.median(durations), 4),
        "p95_ms": round(percentile(durations, 0.95), 4),
        "p99_ms": round(percentile(durations, 0.99), 4),
        "max_ms": round(max(durations), 4),
        "errors": len(errors),
        "error_rate": len(errors) / operations,
        "error_types": sorted(set(errors)),
    }


def operations(
    redis_url: str | None,
) -> tuple[dict[str, Callable[[int], None]], redis.Redis | None]:
    speed = generate_speed_math_challenge(Difficulty.BASIC)
    expected = str(speed.data["expected_answer"])
    signer = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))

    def generate(_index: int) -> None:
        generate_challenge_set(Difficulty.BASIC)

    def verify(_index: int) -> None:
        if not verify_response(speed, expected, 1).passed:
            raise RuntimeError("known-correct benchmark answer failed")

    def sign(index: int) -> None:
        signer.sign(f"mettle-benchmark-{index}".encode("ascii"))

    phases: dict[str, Callable[[int], None]] = {
        "challenge_generation": generate,
        "answer_verification": verify,
        "credential_signing": sign,
    }
    redis_client: redis.Redis | None = None
    if redis_url:
        redis_client = redis.Redis.from_url(
            redis_url, socket_connect_timeout=1, socket_timeout=2
        )
        redis_client.ping()
        contention_key = "mettle:benchmark:contention"
        redis_client.delete(contention_key)

        def contend(_index: int) -> None:
            redis_client.incr(contention_key)

        phases["redis_contention"] = contend
    return phases, redis_client


def check_budget(results: list[dict], budget: dict) -> list[str]:
    failures = []
    for result in results:
        if result["error_rate"] > budget["error_rate_max"]:
            failures.append(f"{result['name']} error rate {result['error_rate']}")
        if result["concurrency"] == budget["representative_concurrency"]:
            limit = budget["p95_ms_max"][result["name"]]
            if result["p95_ms"] > limit:
                failures.append(f"{result['name']} p95 {result['p95_ms']} > {limit}")
        if result["concurrency"] == budget["overload_concurrency"]:
            limit = budget["overload_p99_ms_max"][result["name"]]
            if result["p99_ms"] > limit:
                failures.append(
                    f"{result['name']} overload p99 {result['p99_ms']} > {limit}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", type=int, default=500)
    parser.add_argument("--redis-url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 50 <= args.operations <= 100000:
        parser.error("--operations must be between 50 and 100000")
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    phases, redis_client = operations(args.redis_url)
    results = []
    for concurrency in (
        1,
        budget["representative_concurrency"],
        budget["overload_concurrency"],
    ):
        for name, operation in phases.items():
            results.append(
                measure(
                    name,
                    operation,
                    operations=args.operations,
                    concurrency=concurrency,
                )
            )
    if redis_client is not None:
        redis_client.delete("mettle:benchmark:contention")
        redis_client.close()
    failures = check_budget(results, budget)
    report = {
        "report_schema_version": "1.0",
        "candidate": "working-tree; bind to an immutable SHA before release use",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "budget": budget,
        "results": results,
        "budget_passed": not failures,
        "failures": failures,
        "production_gate": (
            "Repeat against deployed Redis and API workers on the exact candidate."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if failures:
        for failure in failures:
            print(f"BUDGET FAILURE: {failure}", file=sys.stderr)
        return 1
    print("Bounded load regression budget passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
