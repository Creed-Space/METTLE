"""``mettle`` command-line interface.

Runs METTLE challenges locally and emits an unsigned local verification result.

Usage examples::

    mettle verify                 # interactive basic verification (3 challenges)
    mettle verify --full          # interactive full verification (5 challenges)
    mettle verify --suite native  # run a single suite from the registry

Interactive mode is designed for a Becoming Mind to drive programmatically: each
challenge is written to stdout as a single JSON line, and the respondent's answer is
read as a single line from stdin. The per-challenge time limit is enforced by
wall clock.

Exit codes:
    0  METTLE verification threshold met
    1  METTLE verification threshold not met
    2  usage or configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

from mettle.challenge_adapter import SUITE_REGISTRY, ChallengeAdapter
from mettle.challenger import generate_challenge_set
from mettle.models import Difficulty
from mettle.vcp import compute_tier
from mettle.verifier import compute_mettle_result, verify_response

METTLE_VERSION = "0.4.8"
# Suites without a deterministic single-pass CLI path (handled via hosted API).
_LLM_SUITE = "llm-dynamic"


# === Local result construction ===


def _canonical_bytes(claims: dict[str, Any]) -> bytes:
    """Deterministic byte encoding of credential claims for signing."""
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_credential(
    *,
    mode: str,
    scope: str,
    entity_id: str | None,
    suites_passed: list[str],
    suites_failed: list[str],
    tier: str,
    verified: bool,
    pass_rate: float,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Build an unsigned local result without claiming server issuance."""
    issued_at = datetime.now(tz=timezone.utc).isoformat()
    return {
        "mettle_version": METTLE_VERSION,
        "receipt_type": "mettle-local-verification-result",
        "mode": mode,
        "scope": scope,
        "entity_id": entity_id,
        "suites_passed": sorted(suites_passed),
        "suites_failed": sorted(suites_failed),
        "tier": tier,
        "verified": verified,
        "screening_passed": verified,
        "assurance": "mettle_local_behavioral_verification",
        "credential_eligible": False,
        "pass_rate": round(pass_rate, 4),
        "issued_at": issued_at,
        "details": details,
        "signature": None,
    }


def verify_credential(
    credential: dict[str, Any],
    *,
    trusted_keyring: dict[str, str] | None = None,
    status_receipt: dict[str, Any] | None = None,
) -> bool:
    """Accept an issuer credential only against external trust and live status.

    Claimant-supplied public keys are intentionally ignored. Local unsigned
    receipts and historical self-signed envelopes are not trust credentials.
    """
    if trusted_keyring is None or status_receipt is None:
        return False
    from mettle.vcp import verify_mettle_credential_with_status

    return verify_mettle_credential_with_status(
        credential,
        trusted_keyring,
        status_receipt,
    )


# === Quick verification (simple challenge set) ===


def _emit(obj: dict[str, Any]) -> None:
    """Write a single JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def run_quick(
    difficulty: Difficulty,
    *,
    entity_id: str | None,
    quiet: bool,
) -> dict[str, Any]:
    """Run the quick challenge battery, returning a local result."""
    challenges = generate_challenge_set(difficulty)
    results = []

    for challenge in challenges:
        sanitized = challenge.sanitized()
        challenge_line = {
            "kind": "challenge",
            "id": sanitized.id,
            "type": sanitized.type.value,
            "prompt": sanitized.prompt,
            "time_limit_ms": sanitized.time_limit_ms,
            "data": sanitized.data,
        }

        _emit(challenge_line)
        start = time.monotonic()
        answer = sys.stdin.readline()
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if answer == "":  # EOF
            answer = ""
        answer = answer.rstrip("\n")

        result = verify_response(challenge, answer, elapsed_ms)
        results.append(result)
        if not quiet:
            _emit(
                {
                    "kind": "result",
                    "challenge_id": result.challenge_id,
                    "passed": result.passed,
                    "response_time_ms": result.response_time_ms,
                    "time_limit_ms": result.time_limit_ms,
                }
            )

    mettle_result = compute_mettle_result(results, entity_id=entity_id)
    passed_types = sorted({r.challenge_type.value for r in results if r.passed})
    failed_types = sorted({r.challenge_type.value for r in results if not r.passed})
    return build_credential(
        mode="interactive",
        scope=f"quick:{difficulty.value}",
        entity_id=entity_id,
        suites_passed=passed_types,
        suites_failed=failed_types,
        tier=(
            "silver"
            if mettle_result.verified and difficulty == Difficulty.FULL
            else "bronze"
            if mettle_result.verified
            else "none"
        ),
        verified=mettle_result.screening_passed,
        pass_rate=mettle_result.pass_rate,
        details={
            "challenges": [
                {
                    "type": r.challenge_type.value,
                    "passed": r.passed,
                    "response_time_ms": r.response_time_ms,
                    "time_limit_ms": r.time_limit_ms,
                }
                for r in results
            ],
        },
    )


# === Suite verification (ChallengeAdapter bundle) ===


def run_suite(
    suite: str,
    *,
    entity_id: str | None,
    quiet: bool,
) -> dict[str, Any]:
    """Run a single registry suite, returning a local result.

    Raises:
        ValueError: If the suite cannot be run in single-pass CLI mode.
    """
    generators = {
        "adversarial": ChallengeAdapter.generate_adversarial,
        "native": ChallengeAdapter.generate_native,
        "self-reference": ChallengeAdapter.generate_self_reference,
        "social": ChallengeAdapter.generate_social,
        "inverse-turing": ChallengeAdapter.generate_inverse_turing,
        "anti-thrall": ChallengeAdapter.generate_anti_thrall,
        "agency": ChallengeAdapter.generate_agency,
        "counter-coaching": ChallengeAdapter.generate_counter_coaching,
        "intent-provenance": ChallengeAdapter.generate_intent_provenance,
        "governance": ChallengeAdapter.generate_governance,
    }

    gen = generators.get(suite)
    if gen is None:
        raise ValueError(
            f"Suite '{suite}' cannot be run in single-pass CLI mode. "
            "Multi-round and LLM-dynamic suites require the hosted API at "
            "the hosted API."
        )

    client_data, server_answers = gen()
    time_limit_ms = 30_000  # 30s single-shot budget

    challenge_line = {
        "kind": "suite_challenge",
        "suite": suite,
        "time_limit_ms": time_limit_ms,
        "client_data": client_data,
    }

    _emit(challenge_line)
    start = time.monotonic()
    raw = sys.stdin.readline()
    elapsed_ms = int((time.monotonic() - start) * 1000)
    try:
        answers = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON answer for suite '{suite}': {e}") from e

    time_ok = elapsed_ms <= time_limit_ms
    evaluation = ChallengeAdapter.evaluate_single_shot(suite, answers, server_answers)
    passed = bool(evaluation.get("passed", False)) and time_ok

    if not quiet:
        _emit(
            {
                "kind": "result",
                "suite": suite,
                "passed": passed,
                "score": evaluation.get("score"),
            }
        )

    suites_passed = [suite] if passed else []
    suites_failed = [] if passed else [suite]
    tier = compute_tier(suites_passed)

    return build_credential(
        mode="interactive",
        scope=f"suite:{suite}",
        entity_id=entity_id,
        suites_passed=suites_passed,
        suites_failed=suites_failed,
        tier=tier,
        verified=passed,
        pass_rate=1.0 if passed else float(evaluation.get("score", 0.0)),
        details={
            "suite": suite,
            "score": evaluation.get("score"),
            "time_ok": time_ok,
            "elapsed_ms": elapsed_ms,
            "evaluation": evaluation.get("details", {}),
        },
    )


# === Argument parsing / dispatch ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mettle",
        description="METTLE: run reverse-CAPTCHA challenge suites.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mettle {METTLE_VERSION}"
    )
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser(
        "verify", help="Run local verification and emit an unsigned local result."
    )
    mode_group = verify.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--basic",
        action="store_true",
        help="Quick verification (3 challenges). Default.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full verification (5 challenges, strict timing).",
    )
    mode_group.add_argument(
        "--suite",
        metavar="NAME",
        help="Run a single suite from the registry (see `mettle suites`).",
    )
    verify.add_argument(
        "--json", action="store_true", help="Emit only the JSON receipt to stdout."
    )
    verify.add_argument(
        "--entity-id", metavar="ID", help="Optional identifier for this agent."
    )
    sub.add_parser("suites", help="List the available verification suites.")
    return parser


def _print_suites() -> None:
    for name, (display, description, number) in sorted(
        SUITE_REGISTRY.items(), key=lambda kv: kv[1][2]
    ):
        sys.stdout.write(f"{number:>2}. {name:<18} {display} — {description}\n")


def _human_summary(credential: dict[str, Any]) -> None:
    status = (
        "SCREENING PASSED" if credential["screening_passed"] else "SCREENING NOT PASSED"
    )
    sys.stderr.write(
        f"\nMETTLE {status}\n"
        "  assurance: local result; portable credentials require server issuance\n"
        f"  scope:   {credential['scope']}\n"
        f"  tier:    {credential['tier']}\n"
        f"  passed:  {credential['suites_passed']}\n"
        f"  failed:  {credential['suites_failed']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "suites":
        _print_suites()
        return 0

    if args.command != "verify":
        parser.print_help()
        return 2

    # Validate suite name early.
    if args.suite is not None and args.suite not in SUITE_REGISTRY:
        valid = ", ".join(sorted(SUITE_REGISTRY))
        sys.stderr.write(f"Unknown suite: {args.suite!r}. Valid suites: {valid}\n")
        return 2

    # LLM-dynamic guard mirrors llm_challenges_available().
    if args.suite == _LLM_SUITE:
        from mettle.llm_challenges import is_available as llm_available

        if not llm_available():
            sys.stderr.write(
                "Suite 'llm-dynamic' requires ANTHROPIC_API_KEY (and the anthropic package). "
                "Skipping.\n"
            )
            return 2

    try:
        if args.suite is not None:
            credential = run_suite(
                args.suite, entity_id=args.entity_id, quiet=args.json
            )
        else:
            difficulty = Difficulty.FULL if args.full else Difficulty.BASIC
            credential = run_quick(
                difficulty, entity_id=args.entity_id, quiet=args.json
            )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2

    # Emit the credential (always machine-readable on stdout).
    _emit(credential)
    if not args.json:
        _human_summary(credential)

    return 0 if credential["screening_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
