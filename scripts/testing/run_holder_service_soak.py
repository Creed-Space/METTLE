#!/usr/bin/env python3
"""Exercise Vault custody, issuer rotation, PostgreSQL restart, and concurrency."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)

import mettle.signing as issuer_signing
from mettle.holder import (
    FileSecretProvider,
    HolderPolicy,
    HolderPolicyError,
    PresenceHolder,
    VaultTransitEd25519Signer,
)
from mettle.holder_service import (
    HolderServiceUnavailable,
    PersistentHolderRuntime,
    PostgresHolderStateStore,
)
from mettle.presence import (
    key_fingerprint,
    presence_state_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
)
from mettle.vcp import build_mettle_attestation


ISSUER = "https://mettle-holder-soak.example"
AUDIENCE = "holder-soak.example"
BRONZE_SUITES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
]


def _public_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )


def _presence(
    holder: PresenceHolder,
    issuer_key: Ed25519PrivateKey,
    key_id: str,
    *,
    session_id: str,
    nonce: str | None,
    transcript_hash: str,
    sequence: int,
    action: str | None,
    completed: bool,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "protocol": "mettle-presence-v1",
        "key_fingerprint": holder.key_fingerprint,
        "audience": AUDIENCE,
        "nonce": nonce,
        "transcript_hash": transcript_hash,
        "sequence": sequence,
        "action": action,
        "completed": completed,
    }
    state["issuer_receipt"] = {
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature": base64.b64encode(
            issuer_key.sign(
                presence_state_signing_bytes(
                    session_id=session_id,
                    presence=state,
                )
            )
        ).decode("ascii"),
    }
    return state


def _database_url(environment_name: str) -> str:
    value = os.environ.get(environment_name)
    if not value:
        raise SystemExit(f"{environment_name} is unset or empty")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    public_key_pem = Path(args.vault_public_key_file).read_text()
    vault_token_provider = FileSecretProvider(args.vault_token_file)
    state_key = FileSecretProvider(args.state_hmac_key_file)().encode("utf-8")
    signer = VaultTransitEd25519Signer(
        base_url=args.vault_url,
        mount_path=args.vault_mount,
        key_name=args.vault_key,
        public_key_pem=public_key_pem,
        token_provider=vault_token_provider,
        key_version=args.vault_key_version,
        timeout_seconds=args.timeout,
    )
    verifier = load_pem_public_key(public_key_pem.encode("ascii"))
    if not isinstance(verifier, Ed25519PublicKey):
        raise RuntimeError("Vault public key is not Ed25519")
    messages = [f"vault-concurrency-{index}".encode("ascii") for index in range(64)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        signatures = list(executor.map(signer.sign, messages))
    for message, signature in zip(messages, signatures, strict=True):
        verifier.verify(signature, message)

    old_issuer_key = Ed25519PrivateKey.generate()
    new_issuer_key = Ed25519PrivateKey.generate()
    holder_id = args.holder_id or f"holder-soak-{uuid.uuid4().hex}"
    policy = HolderPolicy(
        issuer_public_keys={},
        issuer_public_keyrings={
            ISSUER: {
                "mettle-vcp-old": _public_pem(old_issuer_key),
                "mettle-vcp-new": _public_pem(new_issuer_key),
            }
        },
        allowed_audiences=frozenset({AUDIENCE}),
        max_active_sessions=4,
        max_actions_per_session=4,
        max_presentations_per_credential=128,
        max_session_records=128,
        max_credentials=128,
        max_presentation_records=1024,
    )
    database_url = _database_url(args.database_url_env)
    first_store = PostgresHolderStateStore(database_url, holder_id)
    runtime = PersistentHolderRuntime(
        PresenceHolder(signer, policy), first_store, state_key
    )
    session_id = f"session-{uuid.uuid4().hex}"
    initial_hash = "sha256:" + "a" * 64
    runtime.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(
            runtime.holder,
            old_issuer_key,
            "mettle-vcp-old",
            session_id=session_id,
            nonce="n" * 32,
            transcript_hash=initial_hash,
            sequence=0,
            action="suite:adversarial",
            completed=False,
        ),
    )
    payload_hash = "sha256:" + "b" * 64
    submission_signature = runtime.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    submission_message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    final_hash = transcript_hash_after_submission(
        previous_transcript_hash=initial_hash,
        message=submission_message,
        signature=submission_signature,
    )
    runtime.commit_submission(
        session_id=session_id,
        presence=_presence(
            runtime.holder,
            new_issuer_key,
            "mettle-vcp-new",
            session_id=session_id,
            nonce=None,
            transcript_hash=final_hash,
            sequence=1,
            action=None,
            completed=True,
        ),
    )

    previous_signing_state = (
        issuer_signing._private_key,
        issuer_signing._public_key,
        issuer_signing._key_id,
        issuer_signing._initialized,
    )
    issuer_signing._private_key = new_issuer_key
    issuer_signing._public_key = new_issuer_key.public_key()
    issuer_signing._key_id = "mettle-vcp-new"
    issuer_signing._initialized = True
    now_ms = int(time.time() * 1000)
    try:
        attestation = build_mettle_attestation(
            session_id=session_id,
            difficulty="standard",
            suites_passed=BRONZE_SUITES,
            suites_failed=[],
            pass_rate=1.0,
            subject_id="holder-soak",
            key_id="mettle-vcp-new",
            presence={
                "protocol": "mettle-presence-v1",
                "public_key_pem": runtime.holder.public_key_pem,
                "key_fingerprint": runtime.holder.key_fingerprint,
                "audience": AUDIENCE,
                "credential_jti": uuid.uuid4().hex,
                "transcript_hash": final_hash,
                "sequence": 1,
                "started_at_unix_ms": now_ms - 100,
                "submissions": [
                    {
                        "sequence": 1,
                        "action": "suite:adversarial",
                        "response_time_ms": 100,
                        "accepted_at_unix_ms": now_ms,
                        "transcript_hash": final_hash,
                    }
                ],
            },
        )
    finally:
        (
            issuer_signing._private_key,
            issuer_signing._public_key,
            issuer_signing._key_id,
            issuer_signing._initialized,
        ) = previous_signing_state
    credential_jti = runtime.register_credential(issuer=ISSUER, attestation=attestation)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    presentation_values = {
        "challenge_id": f"presentation-{uuid.uuid4().hex}",
        "nonce": "p" * 32,
        "audience": AUDIENCE,
        "credential_jti": credential_jti,
        "expires_at": expires_at,
    }
    presentation_signature = runtime.sign_presentation(**presentation_values)
    revision_before_restart = runtime.status()["state_revision"]
    runtime.close()

    restarted_store = PostgresHolderStateStore(database_url, holder_id)
    restarted = PersistentHolderRuntime(
        PresenceHolder(signer, policy), restarted_store, state_key
    )
    replay_signature = restarted.sign_presentation(**presentation_values)
    if replay_signature != presentation_signature:
        raise RuntimeError("Presentation signature changed after restart")
    conflicting_values = {**presentation_values, "nonce": "q" * 32}
    try:
        restarted.sign_presentation(**conflicting_values)
    except HolderPolicyError:
        replay_rejected = True
    else:
        replay_rejected = False
        raise RuntimeError("Conflicting presentation replay was accepted")

    try:
        PostgresHolderStateStore(database_url, holder_id)
    except HolderServiceUnavailable:
        split_brain_rejected = True
    else:
        split_brain_rejected = False
        raise RuntimeError("Concurrent holder instance acquired the same lock")

    concurrent_values = {
        **presentation_values,
        "challenge_id": f"concurrent-{uuid.uuid4().hex}",
        "nonce": "r" * 32,
    }
    with ThreadPoolExecutor(max_workers=16) as executor:
        concurrent_signatures = list(
            executor.map(
                lambda _index: restarted.sign_presentation(**concurrent_values),
                range(64),
            )
        )
    if len(set(concurrent_signatures)) != 1:
        raise RuntimeError("Concurrent idempotent signatures diverged")
    final_status = restarted.status()
    restarted.close()
    return {
        "schema": "mettle-holder-service-soak-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holder_id": holder_id,
        "vault": {
            "url": args.vault_url,
            "mount": args.vault_mount,
            "key": args.vault_key,
            "key_version": args.vault_key_version,
            "public_key_fingerprint": key_fingerprint(public_key_pem),
            "distinct_concurrent_signatures_verified": len(signatures),
            "private_key_in_process": False,
        },
        "issuer_rotation": {
            "initial_key_id": "mettle-vcp-old",
            "completion_key_id": "mettle-vcp-new",
            "credential_key_id": attestation["auditor_key_id"],
            "passed": True,
        },
        "persistence": {
            "revision_before_restart": revision_before_restart,
            "revision_after_restart": final_status["state_revision"],
            "presentation_replay_stable": True,
            "conflicting_replay_rejected": replay_rejected,
            "split_brain_rejected": split_brain_rejected,
        },
        "concurrency": {
            "requests": len(concurrent_signatures),
            "unique_signatures": len(set(concurrent_signatures)),
            "passed": True,
        },
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-url", required=True)
    parser.add_argument("--vault-mount", default="transit")
    parser.add_argument("--vault-key", default="mettle-holder")
    parser.add_argument("--vault-key-version", type=int, required=True)
    parser.add_argument("--vault-public-key-file", required=True)
    parser.add_argument("--vault-token-file", required=True)
    parser.add_argument("--state-hmac-key-file", required=True)
    parser.add_argument("--database-url-env", default="METTLE_HOLDER_DATABASE_URL")
    parser.add_argument("--holder-id")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "status": report["status"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
