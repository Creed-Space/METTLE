#!/usr/bin/env python3
"""Generate deterministic public credential fixtures for independent verifiers."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "fixtures" / "credentials" / "v1.json"
sys.path.insert(0, str(ROOT))

from mettle import signing  # noqa: E402
from mettle.protocol import CREDENTIAL_SCHEMA_VERSION, SUITE_POLICY_VERSION  # noqa: E402
from mettle.vcp import (  # noqa: E402
    SUITE_ORDER,
    _canonical_bytes,
    build_credential_status_receipt,
    build_mettle_attestation,
    verify_mettle_credential_with_status,
)

FIXTURE_KEY_ID = "mettle-fixture-v1"
FIXTURE_KEY_SEED = hashlib.sha256(
    b"METTLE public compatibility fixture key v1; never production"
).digest()


def _resign(attestation: dict, private_key: Ed25519PrivateKey) -> None:
    unsigned = copy.deepcopy(attestation)
    unsigned.pop("signature", None)
    attestation["signature"] = "ed25519:" + base64.b64encode(
        private_key.sign(_canonical_bytes(unsigned))
    ).decode("ascii")


def _issue(
    *,
    private_key: Ed25519PrivateKey,
    reviewed_at: datetime,
    session_id: str,
    subject_id: str,
    entity_id: str,
) -> dict:
    signing._private_key = private_key
    signing._public_key = private_key.public_key()
    signing._key_id = FIXTURE_KEY_ID
    signing._initialized = True
    return build_mettle_attestation(
        session_id=session_id,
        subject_id=subject_id,
        entity_id=entity_id,
        difficulty="standard",
        suites_passed=list(SUITE_ORDER)[:5],
        suites_failed=[],
        pass_rate=1.0,
        key_id=FIXTURE_KEY_ID,
        reviewed_at=reviewed_at,
    )


def _status(
    attestation: dict[str, Any],
    *,
    observed_at: datetime,
    revoked: bool = False,
) -> dict[str, Any]:
    return build_credential_status_receipt(
        attestation["metadata"]["jti"],
        revoked=revoked,
        key_id=FIXTURE_KEY_ID,
        observed_at=observed_at,
    )


def build_fixtures() -> dict:
    private_key = Ed25519PrivateKey.from_private_bytes(FIXTURE_KEY_SEED)
    public_der = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    public_pem = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    issued = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
    valid = _issue(
        private_key=private_key,
        reviewed_at=issued,
        session_id="fixture-valid",
        subject_id="subject-fixture",
        entity_id="agent-fixture",
    )
    unicode_credential = _issue(
        private_key=private_key,
        reviewed_at=issued,
        session_id="fixture-unicode",
        subject_id="devenir-心",
        entity_id="agent-café-🤝",
    )
    tampered = copy.deepcopy(valid)
    tampered["metadata"]["tier"] = "platinum"
    future_policy = copy.deepcopy(valid)
    future_policy["metadata"]["suite_policy_version"] = "future-policy"
    future_policy["content_hash"] = (
        "sha256:"
        + hashlib.sha256(_canonical_bytes(future_policy["metadata"])).hexdigest()
    )
    _resign(future_policy, private_key)
    presence_bearer = copy.deepcopy(valid)
    transcript_hash = (
        "sha256:" + hashlib.sha256(b"fixture copied presence bearer").hexdigest()
    )
    presence_bearer["attestation_type"] = "mettle-presence-credential"
    presence_bearer["metadata"]["audience"] = "https://verifier.fixture.test"
    presence_bearer["metadata"]["proof_of_possession"] = {
        "protocol": "mettle-presence-v1",
        "public_key_pem": public_pem,
        "key_fingerprint": "sha256:" + hashlib.sha256(public_der).hexdigest(),
        "transcript_hash": transcript_hash,
        "sequence": 1,
        "server_timing": {
            "total_elapsed_ms": 12,
            "submissions": [
                {
                    "sequence": 1,
                    "action": "suite:adversarial",
                    "response_time_ms": 12,
                    "transcript_hash": transcript_hash,
                }
            ],
        },
    }
    presence_bearer["content_hash"] = (
        "sha256:"
        + hashlib.sha256(_canonical_bytes(presence_bearer["metadata"])).hexdigest()
    )
    _resign(presence_bearer, private_key)

    ordinary_check = issued + timedelta(minutes=30)
    expired_check = issued + timedelta(hours=2)
    cases: list[dict[str, Any]] = [
        {
            "name": "valid-bronze",
            "expected_valid": True,
            "verification_time": ordinary_check.isoformat(),
            "attestation": valid,
            "status_receipt": _status(
                valid, observed_at=ordinary_check - timedelta(minutes=1)
            ),
        },
        {
            "name": "unicode-valid",
            "expected_valid": True,
            "verification_time": ordinary_check.isoformat(),
            "attestation": unicode_credential,
            "status_receipt": _status(
                unicode_credential, observed_at=ordinary_check - timedelta(minutes=1)
            ),
        },
        {
            "name": "tampered-tier",
            "expected_valid": False,
            "verification_time": ordinary_check.isoformat(),
            "attestation": tampered,
            "status_receipt": _status(
                valid, observed_at=ordinary_check - timedelta(minutes=1)
            ),
        },
        {
            "name": "expired",
            "expected_valid": False,
            "verification_time": expired_check.isoformat(),
            "attestation": valid,
            "status_receipt": _status(
                valid, observed_at=expired_check - timedelta(minutes=1)
            ),
        },
        {
            "name": "unsupported-policy",
            "expected_valid": False,
            "verification_time": ordinary_check.isoformat(),
            "attestation": future_policy,
            "status_receipt": _status(
                valid, observed_at=ordinary_check - timedelta(minutes=1)
            ),
        },
        {
            "name": "presence-requires-live-holder-presentation",
            "expected_valid": False,
            "verification_time": ordinary_check.isoformat(),
            "attestation": presence_bearer,
            "status_receipt": _status(
                presence_bearer,
                observed_at=ordinary_check - timedelta(minutes=1),
            ),
        },
        {
            "name": "revoked",
            "expected_valid": False,
            "verification_time": ordinary_check.isoformat(),
            "attestation": valid,
            "status_receipt": _status(
                valid,
                observed_at=ordinary_check - timedelta(minutes=1),
                revoked=True,
            ),
        },
    ]
    keyring = {FIXTURE_KEY_ID: public_pem}
    for case in cases:
        observed = verify_mettle_credential_with_status(
            case["attestation"],
            keyring,
            case["status_receipt"],
            now=datetime.fromisoformat(case["verification_time"]),
        )
        if observed is not case["expected_valid"]:
            raise RuntimeError(f"Fixture self-check failed for {case['name']}")

    return {
        "fixture_schema_version": "1.0",
        "credential_schema_version": CREDENTIAL_SCHEMA_VERSION,
        "suite_policy_version": SUITE_POLICY_VERSION,
        "warning": "Deterministic public test key. Never use this key for issuance.",
        "canonicalization": (
            "Recursively sort object keys, encode UTF-8 JSON without whitespace, "
            "and encode integral numbers as JSON integers."
        ),
        "key": {
            "key_id": FIXTURE_KEY_ID,
            "algorithm": "Ed25519",
            "public_key_pem": public_pem,
            "spki_sha256": "sha256:" + hashlib.sha256(public_der).hexdigest(),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    rendered = (
        json.dumps(build_fixtures(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    if args.update:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(f"Updated {OUTPUT.relative_to(ROOT)}")
        return 0
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
        print(
            "Credential fixtures are stale; review and run "
            "python3 scripts/generate_credential_fixtures.py --update",
            file=sys.stderr,
        )
        return 1
    print("Credential fixtures are deterministic and current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
