"""Security tests for the autonomous Presence holder policy boundary."""

from __future__ import annotations

import base64
import copy
import ssl
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_public_key,
)
from cryptography.x509.oid import NameOID

import mettle.signing as issuer_signing
from mettle.holder import (
    EphemeralEd25519Signer,
    HolderPolicy,
    HolderPolicyError,
    MacOSKeychainEd25519Signer,
    MacOSKeychainSecretProvider,
    PresenceHolder,
    RenewingVaultTokenProvider,
    VaultTransitEd25519Signer,
)
from mettle.presence import (
    presence_state_signing_bytes,
    presentation_signing_bytes,
    submission_signing_bytes,
    transcript_hash_after_submission,
)
from mettle.vcp import build_mettle_attestation


ISSUER = "https://mettle.example"
AUDIENCE = "service.example"
BRONZE_SUITES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
]


def _ca_pem() -> str:
    key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "METTLE test CA")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, algorithm=None)
    )
    return certificate.public_bytes(Encoding.PEM).decode("ascii")


@pytest.fixture()
def issuer_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("METTLE_DEV_MODE", "true")
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False
    assert issuer_signing.init_signing() is True
    public_key_pem = issuer_signing.get_public_key_pem()
    assert isinstance(public_key_pem, str)
    yield public_key_pem
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False


def _presence(
    holder: PresenceHolder,
    *,
    session_id: str = "session-1",
    nonce: str | None = "n" * 32,
    transcript_hash: str = "sha256:" + "a" * 64,
    sequence: int = 0,
    action: str | None = "suite:adversarial",
    completed: bool = False,
    audience: str = AUDIENCE,
    signed: bool = True,
) -> dict[str, object]:
    state: dict[str, object] = {
        "protocol": "mettle-presence-v1",
        "key_fingerprint": holder.key_fingerprint,
        "audience": audience,
        "nonce": nonce,
        "transcript_hash": transcript_hash,
        "sequence": sequence,
        "action": action,
        "completed": completed,
    }
    if signed:
        _sign_presence_state(state, session_id=session_id)
    return state


def _sign_presence_state(state: dict[str, object], *, session_id: str) -> None:
    state["issuer_receipt"] = {
        "key_id": "mettle-vcp-v1",
        "algorithm": "Ed25519",
        "signature": issuer_signing.sign_attestation(
            presence_state_signing_bytes(
                session_id=session_id,
                presence=state,
            )
        ),
    }


def _public_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )


def _sign_presence_state_with_key(
    state: dict[str, object],
    *,
    session_id: str,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> None:
    state["issuer_receipt"] = {
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature": base64.b64encode(
            private_key.sign(
                presence_state_signing_bytes(
                    session_id=session_id,
                    presence=state,
                )
            )
        ).decode("ascii"),
    }


def _holder(
    issuer_public_key_pem: str,
    *,
    max_active_sessions: int = 1,
    max_actions: int = 2,
    max_presentations: int = 2,
) -> tuple[PresenceHolder, EphemeralEd25519Signer]:
    signer = EphemeralEd25519Signer()
    return (
        PresenceHolder(
            signer,
            HolderPolicy(
                issuer_public_keys={ISSUER: issuer_public_key_pem},
                allowed_audiences=frozenset({AUDIENCE}),
                max_active_sessions=max_active_sessions,
                max_actions_per_session=max_actions,
                max_presentations_per_credential=max_presentations,
            ),
        ),
        signer,
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"max_active_sessions": 0}, "Active-session budget"),
        ({"max_actions_per_session": 0}, "Per-session action budget"),
        ({"max_presentations_per_credential": 0}, "Presentation budget"),
        ({"max_presentation_ttl_seconds": 0}, "Presentation TTL"),
        ({"issuer_public_keys": {}}, "trusted issuer"),
        ({"allowed_audiences": frozenset()}, "audience"),
        (
            {"issuer_public_keys": {"http://evil.example": "issuer-key"}},
            "use HTTPS",
        ),
        (
            {"issuer_public_keys": {f"{ISSUER}?query=1": "issuer-key"}},
            "query or fragment",
        ),
    ],
)
def test_holder_validates_static_policy(
    issuer_key: str, updates: dict[str, Any], error: str
) -> None:
    values: dict[str, Any] = {
        "issuer_public_keys": {ISSUER: issuer_key},
        "allowed_audiences": frozenset({AUDIENCE}),
    }
    values.update(updates)
    if "issuer_public_keys" in updates and updates["issuer_public_keys"]:
        values["issuer_public_keys"] = {
            next(iter(updates["issuer_public_keys"])): issuer_key
        }
    with pytest.raises(HolderPolicyError, match=error):
        PresenceHolder(EphemeralEd25519Signer(), HolderPolicy(**values))


@pytest.mark.parametrize(
    "field",
    [
        "max_active_sessions",
        "max_actions_per_session",
        "max_presentations_per_credential",
        "max_presentation_ttl_seconds",
        "max_session_records",
        "max_credentials",
        "max_presentation_records",
    ],
)
def test_every_integer_holder_policy_field_rejects_boolean_values(
    issuer_key: str, field: str
) -> None:
    values: dict[str, Any] = {
        "issuer_public_keys": {ISSUER: issuer_key},
        "allowed_audiences": frozenset({AUDIENCE}),
        field: True,
    }
    with pytest.raises(HolderPolicyError, match="budget|TTL|record"):
        PresenceHolder(EphemeralEd25519Signer(), HolderPolicy(**values))


@pytest.mark.parametrize(
    "updates",
    [
        {"issuer_public_keys": [ISSUER]},
        {"issuer_public_keyrings": [ISSUER]},
        {"allowed_audiences": {AUDIENCE}},
        {"allowed_audiences": AUDIENCE},
    ],
)
def test_holder_policy_requires_declared_trust_collection_types(
    issuer_key: str, updates: dict[str, Any]
) -> None:
    values: dict[str, Any] = {
        "issuer_public_keys": {ISSUER: issuer_key},
        "issuer_public_keyrings": {},
        "allowed_audiences": frozenset({AUDIENCE}),
    }
    values.update(updates)
    with pytest.raises(HolderPolicyError, match="mapping|frozen"):
        PresenceHolder(EphemeralEd25519Signer(), HolderPolicy(**values))


def test_macos_keychain_signer_loads_without_shell_or_plaintext_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        Encoding.PEM,
        PrivateFormat.PKCS8,
        NoEncryption(),
    )
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=private_pem.hex().encode("ascii") + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr("mettle.holder.subprocess.run", fake_run)
    signer = MacOSKeychainEd25519Signer(
        service="mettle-holder",
        account="presence-key",
    )
    message = b"holder-keychain-test"
    signature = signer.sign(message)
    public_key = load_pem_public_key(signer.public_key_pem.encode("ascii"))
    assert isinstance(public_key, Ed25519PublicKey)
    public_key.verify(signature, message)
    assert observed["command"] == [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        "mettle-holder",
        "-a",
        "presence-key",
        "-w",
    ]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "timeout": 5.0,
    }


def test_macos_keychain_signer_hides_lookup_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = b"private key lookup detail must not escape"
    monkeypatch.setattr(
        "mettle.holder.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 44, stdout=b"", stderr=diagnostic
        ),
    )
    with pytest.raises(HolderPolicyError, match="lookup failed") as error:
        MacOSKeychainEd25519Signer(service="mettle-holder", account="missing")
    assert diagnostic.decode("ascii") not in str(error.value)


def test_macos_keychain_secret_provider_does_not_retain_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mettle.holder.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=b"short-lived-vault-token\n", stderr=b""
        ),
    )
    provider = MacOSKeychainSecretProvider(
        service="mettle-vault",
        account="holder-token",
    )
    assert provider() == "short-lived-vault-token"
    assert "short-lived-vault-token" not in repr(vars(provider))


def test_renewing_vault_token_provider_renews_once_per_lease_and_on_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ca_file = tmp_path / "vault-ca.pem"
    ca_file.write_text(_ca_pem())
    ca_file.chmod(0o600)
    tokens = ["first-vault-token"]
    clock = [100.0]
    observed: list[dict[str, Any]] = []

    @contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        observed.append({"method": method, "url": url, "kwargs": kwargs})
        yield httpx.Response(
            200,
            json={"auth": {"renewable": True, "lease_duration": 120}},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", fake_stream)
    provider = RenewingVaultTokenProvider(
        base_url="https://vault.example",
        token_provider=lambda: tokens[0],
        ca_file=str(ca_file),
        clock=lambda: clock[0],
    )
    assert provider() == "first-vault-token"
    assert provider() == "first-vault-token"
    assert provider.seconds_until_renewal() == 60.0
    assert len(observed) == 1
    assert observed[0]["method"] == "POST"
    assert observed[0]["url"] == "https://vault.example/v1/auth/token/renew-self"
    assert isinstance(observed[0]["kwargs"]["verify"], ssl.SSLContext)
    assert observed[0]["kwargs"]["follow_redirects"] is False
    assert observed[0]["kwargs"]["trust_env"] is False

    tokens[0] = "rotated-vault-token"
    assert provider() == "rotated-vault-token"
    clock[0] = 161.0
    assert provider.seconds_until_renewal() == 0.0
    assert len(observed) == 2
    retained = repr(vars(provider))
    assert "first-vault-token" not in retained
    assert "rotated-vault-token" not in retained


def test_renewing_vault_token_provider_fails_closed_and_hides_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = "sensitive Vault renewal response"

    @contextmanager
    def failed_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        yield httpx.Response(
            403,
            content=diagnostic,
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", failed_stream)
    provider = RenewingVaultTokenProvider(
        base_url="https://vault.example",
        token_provider=lambda: "vault-token",
    )
    with pytest.raises(HolderPolicyError, match="renewal failed") as error:
        provider()
    assert diagnostic not in str(error.value)


def test_vault_holder_policies_are_renewable_and_least_privilege() -> None:
    vault_dir = Path(__file__).resolve().parents[1] / "deploy" / "vault"
    assert (vault_dir / "mettle-holder-sign.hcl").read_text() == (
        'path "transit/sign/mettle-holder" {\n'
        '  capabilities = ["update"]\n'
        "}\n\n"
        'path "auth/token/renew-self" {\n'
        '  capabilities = ["update"]\n'
        "}\n"
    )
    assert (vault_dir / "mettle-holder-rotate.hcl").read_text() == (
        'path "transit/keys/mettle-holder" {\n'
        '  capabilities = ["read"]\n'
        "}\n\n"
        'path "transit/keys/mettle-holder/rotate" {\n'
        '  capabilities = ["update"]\n'
        "}\n\n"
        'path "auth/token/renew-self" {\n'
        '  capabilities = ["update"]\n'
        "}\n"
    )


def test_vault_transit_signer_keeps_private_key_out_of_process_and_verifies_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault_private_key = Ed25519PrivateKey.generate()
    ca_file = tmp_path / "vault-ca.pem"
    ca_file.write_text(_ca_pem())
    ca_file.chmod(0o600)
    observed: dict[str, Any] = {}

    @contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        observed["method"] = method
        observed["url"] = url
        observed["kwargs"] = kwargs
        message = base64.b64decode(kwargs["json"]["input"], validate=True)
        signature = base64.b64encode(vault_private_key.sign(message)).decode("ascii")
        yield httpx.Response(
            200,
            json={"data": {"signature": f"vault:v7:{signature}"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", fake_stream)
    token_calls = 0

    def token_provider() -> str:
        nonlocal token_calls
        token_calls += 1
        return "test-vault-token"

    signer = VaultTransitEd25519Signer(
        base_url="https://vault.example",
        mount_path="transit",
        key_name="mettle-holder",
        public_key_pem=_public_pem(vault_private_key),
        token_provider=token_provider,
        key_version=7,
        ca_file=str(ca_file),
    )
    message = b"non-exportable-holder-signature"
    signature = signer.sign(message)
    vault_private_key.public_key().verify(signature, message)
    assert token_calls == 1
    assert observed["method"] == "POST"
    assert observed["url"] == ("https://vault.example/v1/transit/sign/mettle-holder")
    assert observed["kwargs"]["headers"] == {"X-Vault-Token": "test-vault-token"}
    assert observed["kwargs"]["follow_redirects"] is False
    assert observed["kwargs"]["trust_env"] is False
    assert isinstance(observed["kwargs"]["verify"], ssl.SSLContext)
    assert base64.b64decode(observed["kwargs"]["json"]["input"]) == message
    assert observed["kwargs"]["json"]["key_version"] == 7
    assert "_private_key" not in vars(signer)
    assert "test-vault-token" not in repr(vars(signer))


def test_vault_transit_signer_fails_closed_on_bad_signature_and_hides_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_key = Ed25519PrivateKey.generate()
    wrong_key = Ed25519PrivateKey.generate()
    diagnostic = "sensitive vault diagnostic"
    signature = base64.b64encode(wrong_key.sign(b"message")).decode("ascii")

    @contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        yield httpx.Response(
            200,
            json={
                "data": {"signature": f"vault:v1:{signature}"},
                "warnings": [diagnostic],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", fake_stream)
    signer = VaultTransitEd25519Signer(
        base_url="https://vault.example",
        mount_path="transit",
        key_name="mettle-holder",
        public_key_pem=_public_pem(expected_key),
        token_provider=lambda: "vault-token",
    )
    with pytest.raises(HolderPolicyError, match="invalid signature") as error:
        signer.sign(b"message")
    assert diagnostic not in str(error.value)


def test_vault_transit_signer_rejects_an_unexpected_key_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()

    @contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        message = base64.b64decode(kwargs["json"]["input"], validate=True)
        signature = base64.b64encode(private_key.sign(message)).decode("ascii")
        yield httpx.Response(
            200,
            json={"data": {"signature": f"vault:v8:{signature}"}},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", fake_stream)
    signer = VaultTransitEd25519Signer(
        base_url="https://vault.example",
        mount_path="transit",
        key_name="mettle-holder",
        public_key_pem=_public_pem(private_key),
        token_provider=lambda: "vault-token",
        key_version=7,
    )
    with pytest.raises(HolderPolicyError, match="unexpected key version"):
        signer.sign(b"message")


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"base_url": "http://vault.example"}, "HTTPS origin"),
        ({"mount_path": "bad/path"}, "mount"),
        ({"key_name": "*"}, "key"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"timeout_seconds": float("nan")}, "timeout"),
        ({"timeout_seconds": True}, "timeout"),
        ({"key_version": 0}, "key version"),
        ({"key_version": True}, "key version"),
        ({"ca_file": "relative-ca.pem"}, "CA file must be absolute"),
        ({"token_provider": cast(Any, None)}, "callable"),
    ],
)
def test_vault_transit_signer_rejects_unsafe_configuration(
    override: dict[str, Any],
    error: str,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    values: dict[str, Any] = {
        "base_url": "https://vault.example",
        "mount_path": "transit",
        "key_name": "mettle-holder",
        "public_key_pem": _public_pem(private_key),
        "token_provider": lambda: "vault-token",
    }
    values.update(override)
    with pytest.raises(HolderPolicyError, match=error):
        VaultTransitEd25519Signer(**values)


def test_vault_transit_signer_sanitizes_transport_and_token_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()

    def build(token_provider: Any = lambda: "vault-token") -> VaultTransitEd25519Signer:
        return VaultTransitEd25519Signer(
            base_url="https://vault.example",
            mount_path="transit",
            key_name="mettle-holder",
            public_key_pem=_public_pem(private_key),
            token_provider=token_provider,
        )

    with pytest.raises(HolderPolicyError, match="empty or oversized"):
        build().sign(b"")
    with pytest.raises(HolderPolicyError, match="token lookup failed"):
        build(lambda: 1 / 0).sign(b"message")
    with pytest.raises(HolderPolicyError, match="token is invalid"):
        build(lambda: "bad\ntoken").sign(b"message")

    @contextmanager
    def failed_stream(method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        raise httpx.ConnectError("sensitive transport diagnostic")
        yield httpx.Response(500)  # pragma: no cover

    monkeypatch.setattr("mettle.holder.httpx.stream", failed_stream)
    with pytest.raises(HolderPolicyError, match="request failed") as error:
        build().sign(b"message")
    assert "sensitive transport diagnostic" not in str(error.value)


def test_vault_transit_signer_bounds_response_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()

    @contextmanager
    def oversized_stream(
        method: str, url: str, **kwargs: Any
    ) -> Iterator[httpx.Response]:
        yield httpx.Response(
            200,
            content=b"x" * 32769,
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("mettle.holder.httpx.stream", oversized_stream)
    signer = VaultTransitEd25519Signer(
        base_url="https://vault.example",
        mount_path="transit",
        key_name="mettle-holder",
        public_key_pem=_public_pem(private_key),
        token_provider=lambda: "vault-token",
    )
    with pytest.raises(HolderPolicyError, match="request failed"):
        signer.sign(b"message")


def test_holder_rejects_ambiguous_or_invalid_issuer_keyrings() -> None:
    issuer_key = _public_pem(Ed25519PrivateKey.generate())
    signer = EphemeralEd25519Signer()
    with pytest.raises(HolderPolicyError, match="both a legacy key"):
        PresenceHolder(
            signer,
            HolderPolicy(
                issuer_public_keys={ISSUER: issuer_key},
                issuer_public_keyrings={f"{ISSUER}/": {"mettle-vcp-v1": issuer_key}},
                allowed_audiences=frozenset({AUDIENCE}),
            ),
        )
    with pytest.raises(HolderPolicyError, match="key_id is invalid"):
        PresenceHolder(
            signer,
            HolderPolicy(
                issuer_public_keys={},
                issuer_public_keyrings={ISSUER: {"bad key id": issuer_key}},
                allowed_audiences=frozenset({AUDIENCE}),
            ),
        )


def test_holder_accepts_trusted_issuer_rotation_during_active_session() -> None:
    old_key = Ed25519PrivateKey.generate()
    new_key = Ed25519PrivateKey.generate()
    signer = EphemeralEd25519Signer()
    holder = PresenceHolder(
        signer,
        HolderPolicy(
            issuer_public_keys={},
            issuer_public_keyrings={
                ISSUER: {
                    "mettle-vcp-2026-01": _public_pem(old_key),
                    "mettle-vcp-2026-02": _public_pem(new_key),
                }
            },
            allowed_audiences=frozenset({AUDIENCE}),
        ),
    )
    session_id = "rotating-issuer-session"
    initial = _presence(holder, session_id=session_id, signed=False)
    _sign_presence_state_with_key(
        initial,
        session_id=session_id,
        key_id="mettle-vcp-2026-01",
        private_key=old_key,
    )
    holder.authorize_session(issuer=ISSUER, session_id=session_id, presence=initial)
    payload_hash = "sha256:" + "b" * 64
    holder_signature = holder.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    transitioned = _presence(
        holder,
        session_id=session_id,
        nonce="o" * 32,
        transcript_hash=transcript_hash_after_submission(
            previous_transcript_hash="sha256:" + "a" * 64,
            message=message,
            signature=holder_signature,
        ),
        sequence=1,
        action="suite:native",
        signed=False,
    )
    _sign_presence_state_with_key(
        transitioned,
        session_id=session_id,
        key_id="mettle-vcp-2026-02",
        private_key=new_key,
    )
    untrusted = copy.deepcopy(transitioned)
    untrusted["issuer_receipt"] = {
        **cast(dict[str, object], untrusted["issuer_receipt"]),
        "key_id": "untrusted-key",
    }
    with pytest.raises(HolderPolicyError, match="not trusted"):
        holder.commit_submission(session_id=session_id, presence=untrusted)
    holder.commit_submission(session_id=session_id, presence=transitioned)
    assert holder.status()["active_sessions"] == 1


def test_holder_rejects_malformed_session_and_presentation_inputs(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    with pytest.raises(HolderPolicyError, match="object"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session",
            presence=cast(Any, None),
        )
    invalid_states: list[tuple[dict[str, object], str]] = [
        ({"protocol": "other"}, "protocol"),
        ({"key_fingerprint": "other"}, "different key"),
        ({"completed": True}, "initial"),
        ({"nonce": "short"}, "at least 32"),
        ({"transcript_hash": "bad"}, "SHA-256"),
        ({"action": "unsupported"}, "supported Presence action"),
    ]
    for changes, error in invalid_states:
        session_id = f"session-{error}"
        presence = _presence(holder, session_id=session_id, signed=False)
        presence.update(changes)
        _sign_presence_state(presence, session_id=session_id)
        with pytest.raises(HolderPolicyError, match=error):
            holder.authorize_session(
                issuer=ISSUER,
                session_id=session_id,
                presence=presence,
            )
    with pytest.raises(HolderPolicyError, match="not authorized"):
        holder.sign_submission(
            session_id="missing",
            action="suite:adversarial",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
    for expires_at, error in [
        ("not-a-date", "expiry is invalid"),
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "expired"),
        ((datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), "exceeds"),
    ]:
        with pytest.raises(HolderPolicyError, match=error):
            holder.sign_presentation(
                challenge_id="challenge",
                nonce="p" * 32,
                audience=AUDIENCE,
                credential_jti="c" * 32,
                expires_at=expires_at,
            )


def test_holder_rejects_unsigned_and_tampered_issuer_state(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key, max_active_sessions=2)
    with pytest.raises(HolderPolicyError, match="issuer receipt is required"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="fabricated-session",
            presence=_presence(
                holder,
                session_id="fabricated-session",
                signed=False,
            ),
        )
    tampered_initial = _presence(holder, session_id="tampered-session")
    tampered_initial["action"] = "suite:native"
    with pytest.raises(HolderPolicyError, match="issuer receipt is invalid"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="tampered-session",
            presence=tampered_initial,
        )

    session_id = "transition-session"
    holder.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(holder, session_id=session_id),
    )
    payload_hash = "sha256:" + "b" * 64
    signature = holder.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=payload_hash,
    )
    next_state = _presence(
        holder,
        session_id=session_id,
        nonce="o" * 32,
        transcript_hash=transcript_hash_after_submission(
            previous_transcript_hash="sha256:" + "a" * 64,
            message=message,
            signature=signature,
        ),
        sequence=1,
        action="suite:native",
    )
    next_state["nonce"] = "p" * 32
    with pytest.raises(HolderPolicyError, match="issuer receipt is invalid"):
        holder.commit_submission(session_id=session_id, presence=next_state)


def test_holder_rejects_untrusted_issuer_audience_and_session_farming(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    with pytest.raises(HolderPolicyError, match="not trusted"):
        holder.authorize_session(
            issuer="https://evil.example",
            session_id="session-evil",
            presence=_presence(holder, session_id="session-evil"),
        )
    with pytest.raises(HolderPolicyError, match="Audience"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session-wrong-audience",
            presence=_presence(
                holder,
                session_id="session-wrong-audience",
                audience="other.example",
            ),
        )
    holder.authorize_session(
        issuer=ISSUER, session_id="session-1", presence=_presence(holder)
    )
    with pytest.raises(HolderPolicyError, match="already"):
        holder.authorize_session(
            issuer=ISSUER, session_id="session-1", presence=_presence(holder)
        )
    with pytest.raises(HolderPolicyError, match="budget"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="session-2",
            presence=_presence(holder, session_id="session-2"),
        )


def test_holder_enforces_pending_payload_and_monotonic_transcript(
    issuer_key: str,
) -> None:
    holder, _ = _holder(issuer_key)
    holder.authorize_session(
        issuer=ISSUER, session_id="session-1", presence=_presence(holder)
    )
    with pytest.raises(HolderPolicyError, match="Action"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:native",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "b" * 64,
        )
    signature = holder.sign_submission(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    assert signature == holder.sign_submission(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    with pytest.raises(HolderPolicyError, match="different submission"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:adversarial",
            nonce="n" * 32,
            previous_transcript_hash="sha256:" + "a" * 64,
            payload_hash="sha256:" + "c" * 64,
        )
    message = submission_signing_bytes(
        session_id="session-1",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    next_hash = transcript_hash_after_submission(
        previous_transcript_hash="sha256:" + "a" * 64,
        message=message,
        signature=signature,
    )
    with pytest.raises(HolderPolicyError, match="advance exactly once"):
        holder.commit_submission(
            session_id="session-1",
            presence=_presence(holder, transcript_hash=next_hash, sequence=0),
        )
    with pytest.raises(HolderPolicyError, match="invalid transcript"):
        holder.commit_submission(
            session_id="session-1",
            presence=_presence(
                holder,
                nonce="o" * 32,
                transcript_hash="sha256:" + "d" * 64,
                sequence=1,
                action="suite:native",
            ),
        )
    holder.commit_submission(
        session_id="session-1",
        presence=_presence(
            holder,
            nonce="o" * 32,
            transcript_hash=next_hash,
            sequence=1,
            action="suite:native",
        ),
    )
    with pytest.raises(HolderPolicyError, match="Nonce"):
        holder.sign_submission(
            session_id="session-1",
            action="suite:native",
            nonce="n" * 32,
            previous_transcript_hash=next_hash,
            payload_hash="sha256:" + "e" * 64,
        )


def test_holder_registers_only_matching_signed_credential_and_bounds_presentation(
    issuer_key: str,
) -> None:
    signer = EphemeralEd25519Signer()
    holder = PresenceHolder(
        signer,
        HolderPolicy(
            issuer_public_keys={},
            issuer_public_keyrings={
                ISSUER: {"mettle-vcp-v1": issuer_key},
            },
            allowed_audiences=frozenset({AUDIENCE}),
            max_actions_per_session=1,
            max_presentations_per_credential=1,
        ),
    )
    session_id = "session-credential"
    initial_hash = "sha256:" + "a" * 64
    holder.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(
            holder,
            session_id=session_id,
            transcript_hash=initial_hash,
        ),
    )
    payload_hash = "sha256:" + "b" * 64
    signature = holder.sign_submission(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash=initial_hash,
        payload_hash=payload_hash,
    )
    final_hash = transcript_hash_after_submission(
        previous_transcript_hash=initial_hash,
        message=message,
        signature=signature,
    )
    holder.commit_submission(
        session_id=session_id,
        presence=_presence(
            holder,
            session_id=session_id,
            nonce=None,
            transcript_hash=final_hash,
            sequence=1,
            action=None,
            completed=True,
        ),
    )
    now_ms = int(time.time() * 1000)
    internal_presence = {
        "protocol": "mettle-presence-v1",
        "public_key_pem": holder.public_key_pem,
        "key_fingerprint": holder.key_fingerprint,
        "audience": AUDIENCE,
        "credential_jti": "c" * 32,
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
    }
    attestation = build_mettle_attestation(
        session_id=session_id,
        difficulty="standard",
        suites_passed=BRONZE_SUITES,
        suites_failed=[],
        pass_rate=1.0,
        subject_id="holder-policy-test",
        presence=internal_presence,
    )
    tampered = copy.deepcopy(attestation)
    tampered["metadata"]["tier"] = "platinum"
    with pytest.raises(HolderPolicyError, match="signature or policy"):
        holder.register_credential(issuer=ISSUER, attestation=tampered)
    assert (
        holder.register_credential(issuer=ISSUER, attestation=attestation) == "c" * 32
    )

    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    presentation_signature = holder.sign_presentation(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    assert presentation_signature == holder.sign_presentation(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    presentation_message = presentation_signing_bytes(
        challenge_id="challenge-1",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti="c" * 32,
        expires_at=expires_at,
    )
    public_key = load_pem_public_key(signer.public_key_pem.encode("ascii"))
    assert isinstance(public_key, Ed25519PublicKey)
    public_key.verify(base64.b64decode(presentation_signature), presentation_message)
    with pytest.raises(HolderPolicyError, match="reused inconsistently"):
        holder.sign_presentation(
            challenge_id="challenge-1",
            nonce="q" * 32,
            audience=AUDIENCE,
            credential_jti="c" * 32,
            expires_at=expires_at,
        )
    with pytest.raises(HolderPolicyError, match="budget"):
        holder.sign_presentation(
            challenge_id="challenge-2",
            nonce="q" * 32,
            audience=AUDIENCE,
            credential_jti="c" * 32,
            expires_at=expires_at,
        )
