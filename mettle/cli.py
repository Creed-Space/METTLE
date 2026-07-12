"""``mettle`` command-line interface.

Runs METTLE verification locally and emits a self-signed credential.

Usage examples::

    mettle verify                 # interactive basic verification (3 challenges)
    mettle verify --full          # interactive full verification (5 challenges)
    mettle verify --auto --json   # auto-solve and print a JSON credential
    mettle verify --suite native  # run a single suite from the registry
    mettle verify --notarize      # (not yet available -- no hosted endpoint)

Interactive mode is designed for an AI agent to drive programmatically: each
challenge is written to stdout as a single JSON line, and the agent's answer is
read as a single line from stdin. The per-challenge time limit is enforced by
wall clock.

Exit codes:
    0  verification passed
    1  verification failed (ran cleanly, did not meet threshold)
    2  usage or configuration error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from mettle.challenge_adapter import SUITE_REGISTRY, ChallengeAdapter
from mettle.challenger import generate_challenge_set
from mettle.models import Difficulty
from mettle.signing import (
    CLI_KEY_ID,
    load_or_create_cli_keypair,
    sign_bytes,
)
from mettle.solver import solve_challenge, solve_suite
from mettle.vcp import compute_tier
from mettle.verifier import compute_mettle_result, verify_response

METTLE_VERSION = "0.1.0"
DEFAULT_API_URL = "https://mettle.sh"

# Suites without a deterministic single-pass CLI path (handled via hosted API).
_LLM_SUITE = "llm-dynamic"


# === Credential construction ===


def _canonical_bytes(claims: dict[str, Any]) -> bytes:
    """Deterministic byte encoding of credential claims for signing."""
    return json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _quick_tier(difficulty: Difficulty, verified: bool) -> str:
    """Map a quick-verification outcome onto a coarse tier.

    Quick verification exercises the fast challenge battery rather than the full
    suite matrix, so it certifies only the entry tiers: basic -> bronze,
    full -> silver. Suite-based runs use :func:`compute_tier` directly.
    """
    if not verified:
        return "none"
    return "silver" if difficulty == Difficulty.FULL else "bronze"


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
    """Build and sign a self-signed METTLE credential."""
    private_key, public_pem = load_or_create_cli_keypair()

    issued_at = datetime.now(tz=timezone.utc).isoformat()
    claims: dict[str, Any] = {
        "mettle_version": METTLE_VERSION,
        "credential_type": "mettle-self-signed",
        "mode": mode,
        "scope": scope,
        "entity_id": entity_id,
        "suites_passed": sorted(suites_passed),
        "suites_failed": sorted(suites_failed),
        "tier": tier,
        "verified": verified,
        "pass_rate": round(pass_rate, 4),
        "issued_at": issued_at,
        "key_id": CLI_KEY_ID,
        "details": details,
    }

    signature = sign_bytes(private_key, _canonical_bytes(claims))

    credential = dict(claims)
    credential["public_key_pem"] = public_pem
    credential["signature"] = f"ed25519:{signature}"
    return credential


def verify_credential(credential: dict[str, Any]) -> bool:
    """Verify a credential's signature against its embedded public key."""
    from mettle.signing import verify_signature

    cred = dict(credential)
    public_pem = cred.pop("public_key_pem", None)
    signature = cred.pop("signature", None)
    if not public_pem or not signature:
        return False
    if signature.startswith("ed25519:"):
        signature = signature[len("ed25519:") :]
    return verify_signature(public_pem, _canonical_bytes(cred), signature)


# === Quick verification (simple challenge set) ===


def _emit(obj: dict[str, Any]) -> None:
    """Write a single JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def run_quick(
    difficulty: Difficulty,
    *,
    auto: bool,
    entity_id: str | None,
    quiet: bool,
) -> dict[str, Any]:
    """Run the quick challenge battery, returning a signed credential."""
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

        if auto:
            start = time.monotonic()
            answer = solve_challenge(
                {
                    "type": sanitized.type.value,
                    "prompt": sanitized.prompt,
                    "data": sanitized.data,
                }
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
        else:
            _emit(challenge_line)
            start = time.monotonic()
            answer = sys.stdin.readline()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if answer == "":  # EOF
                answer = ""
            answer = answer.rstrip("\n")

        result = verify_response(challenge, answer, elapsed_ms)
        results.append(result)
        if not quiet and not auto:
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
    tier = _quick_tier(difficulty, mettle_result.verified)

    return build_credential(
        mode="auto" if auto else "interactive",
        scope=f"quick:{difficulty.value}",
        entity_id=entity_id,
        suites_passed=passed_types,
        suites_failed=failed_types,
        tier=tier,
        verified=mettle_result.verified,
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
    auto: bool,
    entity_id: str | None,
    quiet: bool,
) -> dict[str, Any]:
    """Run a single registry suite, returning a signed credential.

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
            f"{DEFAULT_API_URL}."
        )

    client_data, server_answers = gen()
    time_limit_ms = 30_000  # 30s single-shot budget

    challenge_line = {
        "kind": "suite_challenge",
        "suite": suite,
        "time_limit_ms": time_limit_ms,
        "client_data": client_data,
    }

    if auto:
        start = time.monotonic()
        answers = solve_suite(suite, client_data)
        elapsed_ms = int((time.monotonic() - start) * 1000)
    else:
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

    if not quiet and not auto:
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
        mode="auto" if auto else "interactive",
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


# === Notarization ===


def do_notarize(credential: dict[str, Any], api_key: str | None) -> int:
    """Attempt to notarize a credential with the hosted METTLE service.

    No hosted notarization endpoint currently exists, so this reports that
    notarization is unavailable and returns a non-zero exit code rather than
    inventing an endpoint.
    """
    api_url = os.environ.get("METTLE_API_URL", DEFAULT_API_URL)
    sys.stderr.write(
        "Notarization is not available yet: the hosted METTLE service at "
        f"{api_url} does not expose a credential-notarization endpoint.\n"
        "Your self-signed credential above is still valid for local trust. "
        "Track hosted notarization at https://mettle.sh.\n"
    )
    return 2


# === Argument parsing / dispatch ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mettle",
        description="METTLE: prove you're an AI agent and mint a verifiable credential.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mettle {METTLE_VERSION}"
    )
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser(
        "verify", help="Run a verification and emit a self-signed credential."
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
        "--auto",
        action="store_true",
        help="Auto-solve using the reference solver (demo mode).",
    )
    verify.add_argument(
        "--json", action="store_true", help="Emit only the JSON credential to stdout."
    )
    verify.add_argument(
        "--entity-id", metavar="ID", help="Optional identifier for this agent."
    )
    verify.add_argument(
        "--notarize",
        action="store_true",
        help="Submit the credential to the hosted service for a portable signature.",
    )
    verify.add_argument("--api-key", metavar="KEY", help="API key for --notarize.")

    sub.add_parser("suites", help="List the available verification suites.")
    return parser


def _print_suites() -> None:
    for name, (display, description, number) in sorted(
        SUITE_REGISTRY.items(), key=lambda kv: kv[1][2]
    ):
        sys.stdout.write(f"{number:>2}. {name:<18} {display} — {description}\n")


def _human_summary(credential: dict[str, Any]) -> None:
    status = "VERIFIED" if credential["verified"] else "NOT VERIFIED"
    sys.stderr.write(
        f"\nMETTLE {status}\n"
        f"  scope:   {credential['scope']}\n"
        f"  tier:    {credential['tier']}\n"
        f"  passed:  {credential['suites_passed']}\n"
        f"  failed:  {credential['suites_failed']}\n"
        f"  key_id:  {credential['key_id']}\n"
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
                args.suite, auto=args.auto, entity_id=args.entity_id, quiet=args.json
            )
        else:
            difficulty = Difficulty.FULL if args.full else Difficulty.BASIC
            credential = run_quick(
                difficulty, auto=args.auto, entity_id=args.entity_id, quiet=args.json
            )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2

    # Emit the credential (always machine-readable on stdout).
    _emit(credential)
    if not args.json:
        _human_summary(credential)

    exit_code = 0 if credential["verified"] else 1

    if args.notarize:
        notarize_code = do_notarize(credential, args.api_key)
        if notarize_code != 0:
            return notarize_code

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
