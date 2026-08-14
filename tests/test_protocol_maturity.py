"""Credential version, expiry, and key-rotation acceptance contracts."""

from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone

import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mettle import signing
from mettle.protocol import (
    CREDENTIAL_CLOCK_SKEW_SECONDS,
    CREDENTIAL_SCHEMA_VERSION,
    SUITE_POLICY_VERSION,
    credential_time_window_valid,
)
from mettle.vcp import (
    SUITE_ORDER,
    _canonical_bytes,
    build_mettle_attestation,
    verify_mettle_attestation,
    verify_mettle_attestation_with_keyring,
)


def _public_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )


def _install_signer(monkeypatch, private_key: Ed25519PrivateKey) -> None:
    monkeypatch.setattr(signing, "_private_key", private_key)
    monkeypatch.setattr(signing, "_public_key", private_key.public_key())
    monkeypatch.setattr(signing, "_initialized", True)


def _issue(monkeypatch, private_key, key_id, reviewed_at):
    _install_signer(monkeypatch, private_key)
    return build_mettle_attestation(
        session_id=f"session-{key_id}",
        subject_id="subject",
        difficulty="standard",
        suites_passed=list(SUITE_ORDER)[:5],
        suites_failed=[],
        pass_rate=1.0,
        key_id=key_id,
        reviewed_at=reviewed_at,
    )


def _resign(attestation, private_key):
    unsigned = copy.deepcopy(attestation)
    unsigned.pop("signature", None)
    attestation["signature"] = "ed25519:" + base64.b64encode(
        private_key.sign(_canonical_bytes(unsigned))
    ).decode("ascii")


def test_new_credentials_name_schema_and_suite_policy(monkeypatch):
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)

    assert attestation["metadata"]["credential_schema_version"] == (
        CREDENTIAL_SCHEMA_VERSION
    )
    assert attestation["metadata"]["suite_policy_version"] == SUITE_POLICY_VERSION


def test_current_schema_requires_status_and_legacy_schema_is_rejected(
    monkeypatch,
):
    """Portable acceptance never guesses at retired credential semantics."""
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)
    public_key = _public_pem(key)
    verify_at = reviewed_at + timedelta(minutes=30)

    missing_status = copy.deepcopy(attestation)
    missing_status["metadata"].pop("jti")
    missing_status["metadata"].pop("credential_status")
    missing_status["content_hash"] = (
        "sha256:"
        + __import__("hashlib")
        .sha256(_canonical_bytes(missing_status["metadata"]))
        .hexdigest()
    )
    _resign(missing_status, key)
    assert not verify_mettle_attestation(missing_status, public_key, now=verify_at)

    legacy = copy.deepcopy(missing_status)
    legacy["metadata"]["credential_schema_version"] = "1.0"
    legacy["metadata"]["suite_policy_version"] = "2026-08-12"
    legacy["metadata"].pop("entity_id_binding", None)
    legacy["content_hash"] = (
        "sha256:"
        + __import__("hashlib").sha256(_canonical_bytes(legacy["metadata"])).hexdigest()
    )
    _resign(legacy, key)
    assert not verify_mettle_attestation(legacy, public_key, now=verify_at)


def test_unknown_or_omitted_versions_fail_closed(
    monkeypatch,
):
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)
    public_key = _public_pem(key)
    verify_at = reviewed_at + timedelta(minutes=30)

    historical = copy.deepcopy(attestation)
    historical["metadata"].pop("credential_schema_version")
    historical["metadata"].pop("suite_policy_version")
    historical["content_hash"] = (
        "sha256:"
        + __import__("hashlib")
        .sha256(_canonical_bytes(historical["metadata"]))
        .hexdigest()
    )
    _resign(historical, key)
    assert not verify_mettle_attestation(historical, public_key, now=verify_at)

    future = copy.deepcopy(attestation)
    future["metadata"]["suite_policy_version"] = "future-unknown"
    future["content_hash"] = (
        "sha256:"
        + __import__("hashlib").sha256(_canonical_bytes(future["metadata"])).hexdigest()
    )
    _resign(future, key)
    assert not verify_mettle_attestation(future, public_key, now=verify_at)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda credential: credential.update({"metadata": []}),
        lambda credential: credential["metadata"].update(
            {"suites_passed": [{"nested": "value"}]}
        ),
        lambda credential: credential["metadata"].update(
            {
                "proof_of_possession": {
                    "server_timing": {"submissions": ["not-an-object"]}
                }
            }
        ),
    ],
)
def test_malformed_nested_credential_metadata_fails_closed(monkeypatch, mutation):
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)
    mutation(attestation)

    assert not verify_mettle_attestation(
        attestation,
        _public_pem(key),
        now=reviewed_at + timedelta(minutes=30),
    )


def test_expiry_and_clock_skew_boundaries_are_explicit(monkeypatch):
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)
    public_key = _public_pem(key)
    expires_at = datetime.fromisoformat(attestation["expires_at"])

    assert verify_mettle_attestation(
        attestation,
        public_key,
        now=expires_at - timedelta(microseconds=1),
        clock_skew_seconds=0,
    )
    assert not verify_mettle_attestation(
        attestation, public_key, now=expires_at, clock_skew_seconds=0
    )
    assert verify_mettle_attestation(
        attestation,
        public_key,
        now=expires_at + timedelta(seconds=CREDENTIAL_CLOCK_SKEW_SECONDS - 1),
    )
    assert not verify_mettle_attestation(
        attestation,
        public_key,
        now=expires_at + timedelta(seconds=CREDENTIAL_CLOCK_SKEW_SECONDS),
    )


def test_shared_time_window_rejects_unbounded_skew_and_naive_datetimes():
    """Direct callers cannot bypass the published skew or timezone contract."""
    aware = datetime(2030, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="clock_skew_seconds"):
        credential_time_window_valid(
            reviewed_at=aware,
            expires_at=aware + timedelta(hours=1),
            now=aware,
            clock_skew_seconds=CREDENTIAL_CLOCK_SKEW_SECONDS + 1,
        )

    assert not credential_time_window_valid(
        reviewed_at=aware.replace(tzinfo=None),
        expires_at=aware + timedelta(hours=1),
        now=aware,
    )


def test_recomputed_tier_must_match_signed_claim(monkeypatch):
    key = Ed25519PrivateKey.generate()
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    attestation = _issue(monkeypatch, key, "key-current", reviewed_at)
    public_key = _public_pem(key)

    attestation["metadata"]["tier"] = "silver"
    attestation["content_hash"] = (
        "sha256:"
        + __import__("hashlib")
        .sha256(_canonical_bytes(attestation["metadata"]))
        .hexdigest()
    )
    _resign(attestation, key)

    assert not verify_mettle_attestation(
        attestation, public_key, now=reviewed_at + timedelta(minutes=30)
    )


def test_key_rotation_accepts_overlap_and_rejects_retired_key(monkeypatch):
    reviewed_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    old_key = Ed25519PrivateKey.generate()
    new_key = Ed25519PrivateKey.generate()
    old_credential = _issue(monkeypatch, old_key, "key-old", reviewed_at)
    new_credential = _issue(monkeypatch, new_key, "key-new", reviewed_at)
    verify_at = reviewed_at + timedelta(minutes=30)
    overlap_keyring = {
        "key-old": _public_pem(old_key),
        "key-new": _public_pem(new_key),
    }

    assert verify_mettle_attestation_with_keyring(
        old_credential, overlap_keyring, now=verify_at
    )
    assert verify_mettle_attestation_with_keyring(
        new_credential, overlap_keyring, now=verify_at
    )
    assert not verify_mettle_attestation_with_keyring(
        old_credential, {"key-new": _public_pem(new_key)}, now=verify_at
    )
