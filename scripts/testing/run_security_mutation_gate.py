#!/usr/bin/env python3
"""Prove that focused tests kill security invariant mutations.

This bounded harness is intentionally explicit. It mutates a temporary copy,
never the checkout, and only counts a mutant when its clean baseline test passes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mutation:
    name: str
    invariant: str
    path: str
    original: str
    replacement: str
    test: str


MUTATIONS = (
    Mutation(
        name="mcp-auto-solver-surface-restored",
        invariant="The MCP surface never exposes an automatic challenge solver.",
        path="mettle/mcp_server.py",
        original=(
            "async def list_tools() -> list[Tool]:\n"
            '    """List available METTLE tools."""\n'
            "    return ["
        ),
        replacement=(
            "async def list_tools() -> list[Tool]:\n"
            '    """List available METTLE tools."""\n'
            "    return [\n"
            '        Tool(name="mettle_auto_verify", description="forbidden", '
            'input_schema={"type": "object", "properties": {}}),'
        ),
        test="tests/test_mcp_server.py::test_auto_solver_is_not_exposed",
    ),
    Mutation(
        name="credential-expiry-exclusive-boundary",
        invariant="A credential is invalid exactly at expiry plus allowed skew.",
        path="mettle/protocol.py",
        original="expires_at + skew > current",
        replacement="expires_at + skew >= current",
        test=(
            "tests/test_protocol_maturity.py::"
            "test_expiry_and_clock_skew_boundaries_are_explicit"
        ),
    ),
    Mutation(
        name="current-policy-version-inverted",
        invariant="Only the exact current suite policy version is accepted.",
        path="mettle/protocol.py",
        original="and policy_version in SUPPORTED_SUITE_POLICY_VERSIONS",
        replacement="and policy_version not in SUPPORTED_SUITE_POLICY_VERSIONS",
        test=(
            "tests/test_protocol_maturity.py::"
            "test_unknown_or_omitted_versions_fail_closed"
        ),
    ),
    Mutation(
        name="signed-tier-not-recomputed",
        invariant="The signed tier must equal the tier recomputed from passed suites.",
        path="mettle/vcp.py",
        original="or compute_tier(suites_passed) != tier",
        replacement="or compute_tier(suites_passed) == tier",
        test=(
            "tests/test_protocol_maturity.py::"
            "test_recomputed_tier_must_match_signed_claim"
        ),
    ),
    Mutation(
        name="ed25519-verification-bypass",
        invariant="Altered data must fail Ed25519 verification.",
        path="mettle/signing.py",
        original=(
            "        public_key.verify(base64.b64decode(signature_b64), data)\n"
            "        return True"
        ),
        replacement=(
            "        # SECURITY MUTANT: signature check bypassed\n        return True"
        ),
        test=(
            "tests/test_signing.py::"
            "test_verify_signature_rejects_tampered_data_and_invalid_base64"
        ),
    ),
    Mutation(
        name="presentation-challenge-replay",
        invariant="A successful holder presentation consumes its nonce once.",
        path="mettle/session_manager.py",
        original="            await delete(_presentation_key(challenge_id))",
        replacement=(
            "            # SECURITY MUTANT: presentation challenge remains reusable"
        ),
        test=(
            "tests/test_presence_protocol.py::"
            "test_key_bound_credential_requires_fresh_holder_proof_and_rejects_replay"
        ),
    ),
    Mutation(
        name="hourly-rate-limit-off-by-one",
        invariant="The hourly limiter rejects creation at the configured maximum.",
        path="mettle/session_manager.py",
        original="hourly_count >= MAX_SESSIONS_PER_HOUR",
        replacement="hourly_count > MAX_SESSIONS_PER_HOUR",
        test=(
            "tests/test_session_manager.py::TestHourlyRateLimit::"
            "test_hourly_rate_limit_blocks_at_max"
        ),
    ),
    Mutation(
        name="session-cancellation-ownership-inverted",
        invariant="A different user cannot cancel a session.",
        path="mettle/session_manager.py",
        original='if session["user_id"] != user_id:\n            return False',
        replacement='if session["user_id"] == user_id:\n            return False',
        test="tests/test_v2_api.py::TestCancelSession::test_cancel_wrong_user",
    ),
    Mutation(
        name="cancelled-creation-quota-leak",
        invariant="Task cancellation releases active and hourly quota reservations.",
        path="mettle/session_manager.py",
        original="        except BaseException:",
        replacement="        except Exception:",
        test=(
            "tests/test_session_manager.py::TestHourlyRateLimit::"
            "test_cancelled_creation_releases_active_and_hourly_reservations"
        ),
    ),
)


def _copy_checkout(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "output",
        "playwright-report",
        "test-results",
        "dist",
        "build",
        "*.mp4",
        "*.wav",
    )
    shutil.copytree(ROOT, destination, ignore=ignored)


def _run_test(checkout: Path, test: str, timeout_seconds: int) -> dict:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = __import__("time").perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", test, "-q", "--tb=short"],
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "timed_out": False,
            "duration_seconds": round(__import__("time").perf_counter() - started, 3),
            "output_tail": (completed.stdout + completed.stderr)[-4000:],
        }
    except subprocess.TimeoutExpired as exc:

        def as_text(value: str | bytes | None) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value or ""

        combined = as_text(exc.stdout) + as_text(exc.stderr)
        return {
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(__import__("time").perf_counter() - started, 3),
            "output_tail": combined[-4000:],
        }


def _validate_anchors(checkout: Path) -> None:
    for mutation in MUTATIONS:
        content = (checkout / mutation.path).read_text(encoding="utf-8")
        count = content.count(mutation.original)
        if count != 1:
            raise RuntimeError(
                f"mutation {mutation.name} expected one source anchor, found {count}"
            )


def run_gate(timeout_seconds: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="mettle-security-mutation-") as temp:
        checkout = Path(temp) / "checkout"
        _copy_checkout(checkout)
        _validate_anchors(checkout)
        results = []
        for mutation in MUTATIONS:
            path = checkout / mutation.path
            baseline = _run_test(checkout, mutation.test, timeout_seconds)
            if baseline["returncode"] != 0 or baseline["timed_out"]:
                results.append(
                    {
                        **asdict(mutation),
                        "baseline": baseline,
                        "mutant": None,
                        "killed": False,
                        "classification": "invalid-baseline",
                    }
                )
                continue

            original_content = path.read_text(encoding="utf-8")
            path.write_text(
                original_content.replace(mutation.original, mutation.replacement, 1),
                encoding="utf-8",
            )
            mutant = _run_test(checkout, mutation.test, timeout_seconds)
            path.write_text(original_content, encoding="utf-8")
            killed = mutant["returncode"] not in (0, None) and not mutant["timed_out"]
            results.append(
                {
                    **asdict(mutation),
                    "baseline": baseline,
                    "mutant": mutant,
                    "killed": killed,
                    "classification": (
                        "killed"
                        if killed
                        else "timeout"
                        if mutant["timed_out"]
                        else "survived"
                    ),
                }
            )

    return {
        "report_schema_version": "1.0",
        "candidate": "working-tree; bind to an immutable SHA before release use",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "mutants": results,
        "killed": sum(result["killed"] for result in results),
        "total": len(results),
        "gate_passed": all(result["killed"] for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 10 <= args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 10 and 600")
    report = run_gate(args.timeout_seconds)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(
        f"Security mutation gate: {report['killed']}/{report['total']} killed",
        file=sys.stderr,
    )
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
