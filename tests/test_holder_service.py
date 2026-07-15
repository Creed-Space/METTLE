"""Security and durability tests for the Vault-backed holder service."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

import mettle.signing as issuer_signing
from mettle.holder import (
    EphemeralEd25519Signer,
    FileSecretProvider,
    HolderPolicy,
    HolderPolicyError,
    PresenceHolder,
)
from mettle.holder_service import (
    HolderServiceSettings,
    HolderServiceUnavailable,
    MemoryHolderStateStore,
    PersistentHolderRuntime,
    PostgresHolderStateStore,
    _load_policy,
    _read_bounded_file,
    build_runtime_from_environment,
    create_holder_service,
)
from mettle.presence import (
    presence_state_signing_bytes,
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
STATE_KEY = b"s" * 32
AUTHORIZATION = {"Authorization": "Bearer holder-control-token"}


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


class _FakePostgresCursor:
    def __init__(self, connection: "_FakePostgresConnection") -> None:
        self.connection = connection
        self.result: tuple[Any, ...] | None = None

    def __enter__(self) -> "_FakePostgresCursor":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, query: str, parameters: tuple[Any, ...] = ()) -> None:
        normalized = " ".join(query.split())
        if self.connection.fail_on and self.connection.fail_on in normalized:
            raise RuntimeError("sensitive database failure")
        if "pg_try_advisory_lock" in normalized:
            self.result = (self.connection.locked,)
        elif normalized.startswith("SELECT state_envelope"):
            self.result = (
                (self.connection.envelope, self.connection.revision)
                if self.connection.envelope is not None
                else None
            )
        elif normalized.startswith("INSERT INTO mettle_holder_state"):
            if self.connection.envelope is None:
                self.connection.revision = parameters[1]
                self.connection.envelope = parameters[2].adapted
                self.result = (self.connection.revision,)
            else:
                self.result = None
        elif normalized.startswith("UPDATE mettle_holder_state"):
            if self.connection.revision == parameters[3]:
                self.connection.revision = parameters[0]
                self.connection.envelope = parameters[1].adapted
                self.result = (self.connection.revision,)
            else:
                self.result = None
        elif normalized == "SELECT 1":
            self.result = (1,)
        elif "pg_advisory_unlock" in normalized:
            self.result = (True,)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.result


class _FakePostgresConnection:
    def __init__(self, *, locked: bool = True) -> None:
        self.autocommit = True
        self.closed = 0
        self.locked = locked
        self.envelope: dict[str, Any] | None = None
        self.revision = 0
        self.fail_on: str | None = None
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakePostgresCursor:
        return _FakePostgresCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = 1


@pytest.fixture()
def issuer_key(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    monkeypatch.setenv("METTLE_DEV_MODE", "true")
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._key_id = "mettle-vcp-v1"
    issuer_signing._initialized = False
    assert issuer_signing.init_signing() is True
    public_key_pem = issuer_signing.get_public_key_pem()
    assert isinstance(public_key_pem, str)
    yield public_key_pem
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._key_id = "mettle-vcp-v1"
    issuer_signing._initialized = False


def _holder(
    issuer_public_key: str,
    *,
    private_key: Ed25519PrivateKey | None = None,
    **policy_overrides: Any,
) -> PresenceHolder:
    policy_values: dict[str, Any] = {
        "issuer_public_keys": {},
        "issuer_public_keyrings": {ISSUER: {"mettle-vcp-v1": issuer_public_key}},
        "allowed_audiences": frozenset({AUDIENCE}),
        "max_active_sessions": 16,
        "max_actions_per_session": 16,
        "max_presentations_per_credential": 64,
    }
    policy_values.update(policy_overrides)
    return PresenceHolder(
        EphemeralEd25519Signer(private_key), HolderPolicy(**policy_values)
    )


def _presence(
    holder: PresenceHolder,
    *,
    session_id: str,
    nonce: str | None = "n" * 32,
    transcript_hash: str = "sha256:" + "a" * 64,
    sequence: int = 0,
    action: str | None = "suite:adversarial",
    completed: bool = False,
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
        "key_id": "mettle-vcp-v1",
        "algorithm": "Ed25519",
        "signature": issuer_signing.sign_attestation(
            presence_state_signing_bytes(session_id=session_id, presence=state)
        ),
    }
    return state


def _complete_credential(
    runtime: PersistentHolderRuntime,
    *,
    session_id: str = "durable-session",
) -> str:
    initial_hash = "sha256:" + "a" * 64
    runtime.authorize_session(
        issuer=ISSUER,
        session_id=session_id,
        presence=_presence(runtime.holder, session_id=session_id),
    )
    payload_hash = "sha256:" + "b" * 64
    signature = runtime.sign_submission(
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
    runtime.commit_submission(
        session_id=session_id,
        presence=_presence(
            runtime.holder,
            session_id=session_id,
            nonce=None,
            transcript_hash=final_hash,
            sequence=1,
            action=None,
            completed=True,
        ),
    )
    now_ms = int(time.time() * 1000)
    attestation = build_mettle_attestation(
        session_id=session_id,
        difficulty="standard",
        suites_passed=BRONZE_SUITES,
        suites_failed=[],
        pass_rate=1.0,
        subject_id="holder-service-test",
        presence={
            "protocol": "mettle-presence-v1",
            "public_key_pem": runtime.holder.public_key_pem,
            "key_fingerprint": runtime.holder.key_fingerprint,
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
        },
    )
    return runtime.register_credential(issuer=ISSUER, attestation=attestation)


def test_file_secret_provider_reads_on_demand_without_retaining_secret(
    tmp_path: Path,
) -> None:
    secret_file = tmp_path / "token"
    secret_file.write_text("short-lived-token\n")
    secret_file.chmod(0o600)
    provider = FileSecretProvider(str(secret_file))
    assert provider() == "short-lived-token"
    assert "short-lived-token" not in repr(vars(provider))

    secret_file.chmod(0o644)
    with pytest.raises(HolderPolicyError, match="permissions"):
        provider()
    secret_file.chmod(0o600)
    symlink = tmp_path / "token-link"
    symlink.symlink_to(secret_file)
    with pytest.raises(HolderPolicyError, match="lookup failed"):
        FileSecretProvider(str(symlink))()


def test_runtime_restart_preserves_presentations_and_concurrent_idempotency(
    issuer_key: str,
) -> None:
    holder_private_key = Ed25519PrivateKey.generate()
    store = MemoryHolderStateStore()
    first = PersistentHolderRuntime(
        _holder(issuer_key, private_key=holder_private_key), store, STATE_KEY
    )
    credential_jti = _complete_credential(first)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    first_signature = first.sign_presentation(
        challenge_id="presentation-before-restart",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti=credential_jti,
        expires_at=expires_at,
    )
    first.close()

    restarted = PersistentHolderRuntime(
        _holder(issuer_key, private_key=holder_private_key), store, STATE_KEY
    )
    assert (
        restarted.sign_presentation(
            challenge_id="presentation-before-restart",
            nonce="p" * 32,
            audience=AUDIENCE,
            credential_jti=credential_jti,
            expires_at=expires_at,
        )
        == first_signature
    )
    with pytest.raises(HolderPolicyError, match="reused inconsistently"):
        restarted.sign_presentation(
            challenge_id="presentation-before-restart",
            nonce="q" * 32,
            audience=AUDIENCE,
            credential_jti=credential_jti,
            expires_at=expires_at,
        )

    values = {
        "challenge_id": "concurrent-presentation",
        "nonce": "r" * 32,
        "audience": AUDIENCE,
        "credential_jti": credential_jti,
        "expires_at": expires_at,
    }
    with ThreadPoolExecutor(max_workers=16) as executor:
        signatures = list(
            executor.map(
                lambda _index: restarted.sign_presentation(**values), range(64)
            )
        )
    assert len(set(signatures)) == 1
    assert restarted.status()["presentations"] == 2


def test_authenticated_state_tampering_fails_closed(issuer_key: str) -> None:
    private_key = Ed25519PrivateKey.generate()
    store = MemoryHolderStateStore()
    runtime = PersistentHolderRuntime(
        _holder(issuer_key, private_key=private_key), store, STATE_KEY
    )
    runtime.authorize_session(
        issuer=ISSUER,
        session_id="tamper-session",
        presence=_presence(runtime.holder, session_id="tamper-session"),
    )
    assert store.envelope is not None
    store.envelope["snapshot"]["sessions"][0]["sequence"] = 99
    with pytest.raises(HolderServiceUnavailable, match="Stored holder state"):
        PersistentHolderRuntime(
            _holder(issuer_key, private_key=private_key), store, STATE_KEY
        )


def test_core_restore_rejects_invalid_embedded_signature(issuer_key: str) -> None:
    private_key = Ed25519PrivateKey.generate()
    holder = _holder(issuer_key, private_key=private_key)
    holder.authorize_session(
        issuer=ISSUER,
        session_id="pending-session",
        presence=_presence(holder, session_id="pending-session"),
    )
    holder.sign_submission(
        session_id="pending-session",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    snapshot = holder.export_state()
    snapshot["sessions"][0]["pending"]["signature"] = "A" * 88
    restored = _holder(issuer_key, private_key=private_key)
    with pytest.raises(HolderPolicyError, match="signature is invalid"):
        restored.restore_state(snapshot)


def test_core_restore_rejects_valid_signature_rebound_to_another_session(
    issuer_key: str,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    first = _holder(issuer_key, private_key=private_key)
    first.authorize_session(
        issuer=ISSUER,
        session_id="first-pending-session",
        presence=_presence(first, session_id="first-pending-session"),
    )
    first.sign_submission(
        session_id="first-pending-session",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    second = _holder(issuer_key, private_key=private_key)
    second.authorize_session(
        issuer=ISSUER,
        session_id="second-pending-session",
        presence=_presence(second, session_id="second-pending-session"),
    )
    second.sign_submission(
        session_id="second-pending-session",
        action="suite:adversarial",
        nonce="n" * 32,
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash="sha256:" + "b" * 64,
    )
    snapshot = first.export_state()
    snapshot["sessions"][0]["pending"] = second.export_state()["sessions"][0]["pending"]
    with pytest.raises(HolderPolicyError, match="message is invalid"):
        _holder(issuer_key, private_key=private_key).restore_state(snapshot)


def test_core_restore_rejects_valid_presentation_rebound_to_challenge(
    issuer_key: str,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    runtime = PersistentHolderRuntime(
        _holder(issuer_key, private_key=private_key),
        MemoryHolderStateStore(),
        STATE_KEY,
    )
    credential_jti = _complete_credential(runtime, session_id="binding-session")
    runtime.sign_presentation(
        challenge_id="signed-challenge",
        nonce="p" * 32,
        audience=AUDIENCE,
        credential_jti=credential_jti,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    snapshot = runtime.holder.export_state()
    snapshot["credentials"][0]["presentations"][0]["challenge_id"] = "rebound-challenge"
    with pytest.raises(HolderPolicyError, match="message is invalid"):
        _holder(issuer_key, private_key=private_key).restore_state(snapshot)


def test_persistence_failure_disables_holder_before_returning_signature(
    issuer_key: str,
) -> None:
    class FailingStore(MemoryHolderStateStore):
        def save(self, envelope: dict[str, Any], expected_revision: int) -> int:
            raise HolderServiceUnavailable("database unavailable")

    runtime = PersistentHolderRuntime(_holder(issuer_key), FailingStore(), STATE_KEY)
    with pytest.raises(HolderServiceUnavailable, match="persistence failed"):
        runtime.authorize_session(
            issuer=ISSUER,
            session_id="failed-write",
            presence=_presence(runtime.holder, session_id="failed-write"),
        )
    assert runtime.available is False
    with pytest.raises(HolderServiceUnavailable, match="unavailable"):
        runtime.status()


def test_holder_service_authentication_limits_and_security_headers(
    issuer_key: str,
) -> None:
    runtime = PersistentHolderRuntime(
        _holder(issuer_key), MemoryHolderStateStore(), STATE_KEY
    )
    service = create_holder_service(
        runtime=runtime,
        control_token_provider=lambda: "holder-control-token",
    )
    with TestClient(service) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.headers["cache-control"] == "no-store"
        assert client.get("/v1/status").status_code == 401
        status = client.get("/v1/status", headers=AUTHORIZATION)
        assert status.status_code == 200
        public_key = client.get("/v1/public-key", headers=AUTHORIZATION)
        assert public_key.status_code == 200
        assert "PRIVATE" not in public_key.json()["public_key_pem"]
        invalid = client.post(
            "/v1/sessions/authorize",
            headers=AUTHORIZATION,
            json={"unexpected": True},
        )
        assert invalid.status_code == 400
        oversized = client.post(
            "/v1/sessions/authorize",
            headers={**AUTHORIZATION, "Content-Type": "application/json"},
            content=b"x" * 1048577,
        )
        assert oversized.status_code == 413


def test_holder_service_settings_fail_closed_without_required_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "METTLE_HOLDER_ID",
        "METTLE_HOLDER_VAULT_URL",
        "METTLE_HOLDER_VAULT_KEY_VERSION",
        "METTLE_HOLDER_VAULT_CA_FILE",
        "METTLE_HOLDER_VAULT_PUBLIC_KEY_FILE",
        "METTLE_HOLDER_VAULT_TOKEN_FILE",
        "METTLE_HOLDER_POLICY_FILE",
        "METTLE_HOLDER_CONTROL_TOKEN_FILE",
        "METTLE_HOLDER_STATE_HMAC_KEY_FILE",
        "METTLE_HOLDER_DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(HolderServiceUnavailable, match="METTLE_HOLDER_ID"):
        HolderServiceSettings.from_environment()


@pytest.mark.parametrize("key_version", ["latest", "9999999999"])
def test_holder_service_settings_require_explicit_numeric_vault_key_version(
    monkeypatch: pytest.MonkeyPatch, key_version: str
) -> None:
    values = {
        "METTLE_HOLDER_ID": "holder-v1",
        "METTLE_HOLDER_VAULT_URL": "https://vault.example",
        "METTLE_HOLDER_VAULT_KEY_VERSION": key_version,
        "METTLE_HOLDER_VAULT_CA_FILE": "/etc/secrets/vault-ca",
        "METTLE_HOLDER_VAULT_PUBLIC_KEY_FILE": "/etc/secrets/public-key",
        "METTLE_HOLDER_VAULT_TOKEN_FILE": "/etc/secrets/vault-token",
        "METTLE_HOLDER_POLICY_FILE": "/etc/secrets/policy",
        "METTLE_HOLDER_CONTROL_TOKEN_FILE": "/etc/secrets/control-token",
        "METTLE_HOLDER_STATE_HMAC_KEY_FILE": "/etc/secrets/state-key",
        "METTLE_HOLDER_DATABASE_URL": "postgresql://database.example/mettle",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(HolderServiceUnavailable, match="KEY_VERSION is invalid"):
        HolderServiceSettings.from_environment()


def test_holder_record_budgets_fail_closed(issuer_key: str) -> None:
    holder = _holder(issuer_key, max_session_records=1)
    holder.authorize_session(
        issuer=ISSUER,
        session_id="first-session",
        presence=_presence(holder, session_id="first-session"),
    )
    with pytest.raises(HolderPolicyError, match="record budget"):
        holder.authorize_session(
            issuer=ISSUER,
            session_id="second-session",
            presence=_presence(holder, session_id="second-session"),
        )


def test_postgres_store_persists_revisions_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakePostgresConnection()
    monkeypatch.setattr(
        "mettle.holder_service.psycopg2.connect", lambda *_args, **_kwargs: connection
    )
    store = PostgresHolderStateStore(
        "postgresql://database.example/mettle", "holder-postgres-test"
    )
    assert connection.autocommit is False
    assert store.load() == (None, 0)
    first = {"schema": "envelope", "snapshot": {"sequence": 1}}
    assert store.save(first, 0) == 1
    assert store.load() == (first, 1)
    second = {"schema": "envelope", "snapshot": {"sequence": 2}}
    assert store.save(second, 1) == 2
    assert store.load() == (second, 2)
    assert store.health() is True
    store.close()
    assert connection.closed == 1
    store.close()


def test_postgres_store_fails_closed_on_split_brain_and_database_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked = _FakePostgresConnection(locked=False)
    monkeypatch.setattr(
        "mettle.holder_service.psycopg2.connect", lambda *_args, **_kwargs: locked
    )
    with pytest.raises(HolderServiceUnavailable, match="Another holder instance"):
        PostgresHolderStateStore(
            "postgresql://database.example/mettle", "holder-split-brain"
        )
    assert locked.rollbacks == 1
    assert locked.closed == 1

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("sensitive database diagnostic")

    monkeypatch.setattr("mettle.holder_service.psycopg2.connect", unavailable)
    with pytest.raises(
        HolderServiceUnavailable, match="initialization failed"
    ) as error:
        PostgresHolderStateStore(
            "postgresql://database.example/mettle", "holder-db-failure"
        )
    assert "sensitive database diagnostic" not in str(error.value)


def test_postgres_store_detects_revision_conflict_and_health_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakePostgresConnection()
    monkeypatch.setattr(
        "mettle.holder_service.psycopg2.connect", lambda *_args, **_kwargs: connection
    )
    store = PostgresHolderStateStore(
        "postgresql://database.example/mettle", "holder-conflict-test"
    )
    assert store.save({"snapshot": 1}, 0) == 1
    with pytest.raises(HolderServiceUnavailable, match="revision changed"):
        store.save({"snapshot": 2}, 0)
    connection.fail_on = "SELECT 1"
    assert store.health() is False
    assert connection.rollbacks >= 2
    connection.fail_on = None
    store.close()


def test_environment_runtime_loads_strict_secret_files_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    issuer_key: str,
) -> None:
    files = {
        "ca": _ca_pem(),
        "public": issuer_key,
        "vault-token": "vault-runtime-token",
        "control-token": "holder-control-token",
        "state-key": "s" * 48,
        "policy": json.dumps(
            {
                "issuer_public_keyrings": {ISSUER: {"mettle-vcp-v1": issuer_key}},
                "allowed_audiences": [AUDIENCE],
                "max_active_sessions": 8,
            }
        ),
    }
    paths: dict[str, Path] = {}
    for name, value in files.items():
        path = tmp_path / name
        path.write_text(value)
        path.chmod(0o600)
        paths[name] = path
    environment = {
        "METTLE_HOLDER_ID": "environment-holder-v1",
        "METTLE_HOLDER_VAULT_URL": "https://vault.example",
        "METTLE_HOLDER_VAULT_KEY_VERSION": "7",
        "METTLE_HOLDER_VAULT_CA_FILE": str(paths["ca"]),
        "METTLE_HOLDER_VAULT_PUBLIC_KEY_FILE": str(paths["public"]),
        "METTLE_HOLDER_VAULT_TOKEN_FILE": str(paths["vault-token"]),
        "METTLE_HOLDER_POLICY_FILE": str(paths["policy"]),
        "METTLE_HOLDER_CONTROL_TOKEN_FILE": str(paths["control-token"]),
        "METTLE_HOLDER_STATE_HMAC_KEY_FILE": str(paths["state-key"]),
        "METTLE_HOLDER_DATABASE_URL": "postgresql://database.example/mettle",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    memory_store = MemoryHolderStateStore()
    renewal_calls = 0

    class _RenewingProvider:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self) -> str:
            nonlocal renewal_calls
            renewal_calls += 1
            return "vault-runtime-token"

    monkeypatch.setattr(
        "mettle.holder_service.RenewingVaultTokenProvider", _RenewingProvider
    )
    monkeypatch.setattr(
        "mettle.holder_service.PostgresHolderStateStore",
        lambda *_args, **_kwargs: memory_store,
    )
    runtime, control_provider = build_runtime_from_environment()
    assert runtime.status()["sessions"] == 0
    assert control_provider() == "holder-control-token"
    assert runtime.holder.key_fingerprint.startswith("sha256:")
    assert renewal_calls == 1
    runtime.close()


@pytest.mark.parametrize(
    "invalid_policy",
    [
        {"unsupported": True},
        {
            "issuer_public_keyrings": {},
            "allowed_audiences": [],
            "max_active_sessions": True,
        },
    ],
)
def test_policy_loader_rejects_unknown_and_non_numeric_values(
    tmp_path: Path, invalid_policy: dict[str, Any]
) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(invalid_policy))
    policy.chmod(0o600)
    with pytest.raises(HolderServiceUnavailable, match="unsupported|values"):
        _load_policy(str(policy))


def test_configuration_file_loader_fails_closed_on_unsafe_files(tmp_path: Path) -> None:
    with pytest.raises(HolderServiceUnavailable, match="absolute"):
        _read_bounded_file("relative-policy.json", "Policy", 100)
    with pytest.raises(HolderServiceUnavailable, match="could not be read"):
        _read_bounded_file(str(tmp_path / "missing"), "Policy", 100)

    unsafe = tmp_path / "unsafe"
    unsafe.write_text("value")
    unsafe.chmod(0o666)
    with pytest.raises(HolderServiceUnavailable, match="permissions"):
        _read_bounded_file(str(unsafe), "Policy", 100)

    empty = tmp_path / "empty"
    empty.touch(mode=0o600)
    with pytest.raises(HolderServiceUnavailable, match="permissions"):
        _read_bounded_file(str(empty), "Policy", 100)

    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff")
    binary.chmod(0o600)
    with pytest.raises(HolderServiceUnavailable, match="UTF-8"):
        _read_bounded_file(str(binary), "Policy", 100)


@pytest.mark.parametrize(
    "contents",
    [
        "not-json",
        "[]",
        json.dumps({"issuer_public_keyrings": [], "allowed_audiences": {}}),
        json.dumps(
            {
                "issuer_public_keyrings": {},
                "allowed_audiences": [["unhashable"]],
            }
        ),
    ],
)
def test_policy_loader_sanitizes_malformed_files(tmp_path: Path, contents: str) -> None:
    policy = tmp_path / "malformed-policy.json"
    policy.write_text(contents)
    policy.chmod(0o600)
    with pytest.raises(HolderServiceUnavailable):
        _load_policy(str(policy))


def test_postgres_store_sanitizes_load_and_save_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakePostgresConnection()
    monkeypatch.setattr(
        "mettle.holder_service.psycopg2.connect", lambda *_args, **_kwargs: connection
    )
    store = PostgresHolderStateStore(
        "postgresql://database.example/mettle", "holder-io-failure"
    )
    connection.fail_on = "SELECT state_envelope"
    with pytest.raises(HolderServiceUnavailable, match="load failed") as load_error:
        store.load()
    assert "sensitive database failure" not in str(load_error.value)
    connection.fail_on = "INSERT INTO"
    with pytest.raises(HolderServiceUnavailable, match="save failed") as save_error:
        store.save({"snapshot": 1}, 0)
    assert "sensitive database failure" not in str(save_error.value)
    connection.fail_on = None
    store.close()


def test_postgres_store_rejects_invalid_configuration_and_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HolderServiceUnavailable, match="database URL"):
        PostgresHolderStateStore("", "holder")
    with pytest.raises(HolderServiceUnavailable, match="Holder ID"):
        PostgresHolderStateStore("postgresql://database.example/mettle", "bad id")

    connection = _FakePostgresConnection()
    connection.fail_on = "CREATE TABLE"
    monkeypatch.setattr(
        "mettle.holder_service.psycopg2.connect", lambda *_args, **_kwargs: connection
    )
    with pytest.raises(HolderServiceUnavailable, match="initialization failed"):
        PostgresHolderStateStore(
            "postgresql://database.example/mettle", "holder-setup-failure"
        )
    assert connection.closed == 1


def test_holder_service_routes_all_authenticated_mutations(
    monkeypatch: pytest.MonkeyPatch,
    issuer_key: str,
) -> None:
    runtime = PersistentHolderRuntime(
        _holder(issuer_key), MemoryHolderStateStore(), STATE_KEY
    )
    monkeypatch.setattr(runtime, "authorize_session", lambda **_values: {"ok": True})
    monkeypatch.setattr(runtime, "sign_submission", lambda **_values: "submission")
    monkeypatch.setattr(runtime, "commit_submission", lambda **_values: {"ok": True})
    monkeypatch.setattr(runtime, "register_credential", lambda **_values: "c" * 32)
    monkeypatch.setattr(runtime, "sign_presentation", lambda **_values: "presentation")
    service = create_holder_service(
        runtime=runtime,
        control_token_provider=lambda: "holder-control-token",
    )
    with TestClient(service) as client:
        assert client.post(
            "/v1/sessions/authorize",
            headers=AUTHORIZATION,
            json={"issuer": ISSUER, "session_id": "session", "presence": {}},
        ).json() == {"ok": True}
        assert client.post(
            "/v1/submissions/sign",
            headers=AUTHORIZATION,
            json={
                "session_id": "session",
                "action": "suite:adversarial",
                "nonce": "n" * 32,
                "previous_transcript_hash": "sha256:" + "a" * 64,
                "payload_hash": "sha256:" + "b" * 64,
            },
        ).json() == {"signature": "submission"}
        assert client.post(
            "/v1/submissions/commit",
            headers=AUTHORIZATION,
            json={"session_id": "session", "presence": {}},
        ).json() == {"ok": True}
        assert client.post(
            "/v1/credentials/register",
            headers=AUTHORIZATION,
            json={"issuer": ISSUER, "attestation": {}},
        ).json() == {"credential_jti": "c" * 32}
        assert client.post(
            "/v1/presentations/sign",
            headers=AUTHORIZATION,
            json={
                "challenge_id": "challenge",
                "nonce": "n" * 32,
                "audience": AUDIENCE,
                "credential_jti": "c" * 32,
                "expires_at": "2026-07-14T12:00:00+00:00",
            },
        ).json() == {"signature": "presentation"}


def test_holder_service_sanitizes_control_token_and_health_failures(
    issuer_key: str,
) -> None:
    class UnhealthyStore(MemoryHolderStateStore):
        def health(self) -> bool:
            return False

    def unavailable_control_token() -> str:
        raise RuntimeError("control token lookup failed")

    runtime = PersistentHolderRuntime(_holder(issuer_key), UnhealthyStore(), STATE_KEY)
    unavailable_token = create_holder_service(
        runtime=runtime,
        control_token_provider=unavailable_control_token,
    )
    with TestClient(unavailable_token) as client:
        response = client.get("/v1/status", headers=AUTHORIZATION)
        assert response.status_code == 503
        assert response.json() == {"detail": "Holder service is unavailable"}
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json() == {"status": "unavailable"}
