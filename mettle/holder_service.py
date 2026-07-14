"""Authenticated, durable service boundary for a Vault-backed Presence holder."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, AsyncIterator, Callable, Protocol

import psycopg2
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg2.extras import Json
from pydantic import BaseModel, ConfigDict, Field

from mettle.holder import (
    FileSecretProvider,
    HolderPolicy,
    HolderPolicyError,
    PresenceHolder,
    VaultTransitEd25519Signer,
)


HOLDER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
STATE_ENVELOPE_SCHEMA = "mettle-holder-envelope-v1"
MAX_REQUEST_BYTES = 1048576


class HolderServiceUnavailable(RuntimeError):
    """The durable holder cannot safely process another request."""


class HolderStateStore(Protocol):
    def load(self) -> tuple[dict[str, Any] | None, int]: ...

    def save(self, envelope: dict[str, Any], expected_revision: int) -> int: ...

    def health(self) -> bool: ...

    def close(self) -> None: ...


class MemoryHolderStateStore:
    """Deterministic state store for tests and in-process verification."""

    def __init__(self) -> None:
        self.envelope: dict[str, Any] | None = None
        self.revision = 0
        self._lock = RLock()

    def load(self) -> tuple[dict[str, Any] | None, int]:
        with self._lock:
            copied = (
                json.loads(json.dumps(self.envelope))
                if self.envelope is not None
                else None
            )
            return copied, self.revision

    def save(self, envelope: dict[str, Any], expected_revision: int) -> int:
        with self._lock:
            if expected_revision != self.revision:
                raise HolderServiceUnavailable("Holder state revision changed")
            self.envelope = json.loads(json.dumps(envelope))
            self.revision += 1
            return self.revision

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None


class PostgresHolderStateStore:
    """Single-writer PostgreSQL store with optimistic revisions and an advisory lock."""

    def __init__(self, database_url: str, holder_id: str) -> None:
        if not isinstance(database_url, str) or not database_url:
            raise HolderServiceUnavailable("Holder database URL is required")
        if HOLDER_ID_PATTERN.fullmatch(holder_id) is None:
            raise HolderServiceUnavailable("Holder ID is invalid")
        self._holder_id = holder_id
        self._lock_id = int.from_bytes(
            hashlib.sha256(holder_id.encode("utf-8")).digest()[:8],
            "big",
            signed=True,
        )
        self._lock = RLock()
        try:
            self._connection = psycopg2.connect(
                database_url,
                connect_timeout=5,
                application_name="mettle-holder-service",
            )
            self._connection.autocommit = False
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mettle_holder_state (
                        holder_id TEXT PRIMARY KEY,
                        revision BIGINT NOT NULL CHECK (revision > 0),
                        state_envelope JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (self._lock_id,))
                locked = cursor.fetchone()
                if not locked or locked[0] is not True:
                    raise HolderServiceUnavailable(
                        "Another holder instance owns the persistence lock"
                    )
            self._connection.commit()
        except HolderServiceUnavailable:
            if hasattr(self, "_connection"):
                self._connection.rollback()
                self._connection.close()
            raise
        except Exception:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise HolderServiceUnavailable(
                "Holder database initialization failed"
            ) from None

    def load(self) -> tuple[dict[str, Any] | None, int]:
        with self._lock:
            try:
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT state_envelope, revision "
                        "FROM mettle_holder_state WHERE holder_id = %s",
                        (self._holder_id,),
                    )
                    row = cursor.fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise HolderServiceUnavailable("Holder state load failed") from None
        if row is None:
            return None, 0
        envelope = row[0]
        if not isinstance(envelope, dict) or not isinstance(row[1], int):
            raise HolderServiceUnavailable("Holder state row is invalid")
        return envelope, row[1]

    def save(self, envelope: dict[str, Any], expected_revision: int) -> int:
        next_revision = expected_revision + 1
        with self._lock:
            try:
                with self._connection.cursor() as cursor:
                    if expected_revision == 0:
                        cursor.execute(
                            """
                            INSERT INTO mettle_holder_state
                                (holder_id, revision, state_envelope)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (holder_id) DO NOTHING
                            RETURNING revision
                            """,
                            (self._holder_id, next_revision, Json(envelope)),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE mettle_holder_state
                            SET revision = %s, state_envelope = %s, updated_at = NOW()
                            WHERE holder_id = %s AND revision = %s
                            RETURNING revision
                            """,
                            (
                                next_revision,
                                Json(envelope),
                                self._holder_id,
                                expected_revision,
                            ),
                        )
                    row = cursor.fetchone()
                    if row is None or row[0] != next_revision:
                        raise HolderServiceUnavailable("Holder state revision changed")
                self._connection.commit()
                return next_revision
            except HolderServiceUnavailable:
                self._connection.rollback()
                raise
            except Exception:
                self._connection.rollback()
                raise HolderServiceUnavailable("Holder state save failed") from None

    def health(self) -> bool:
        with self._lock:
            try:
                with self._connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    healthy = cursor.fetchone() == (1,)
                self._connection.commit()
                return healthy
            except Exception:
                self._connection.rollback()
                return False

    def close(self) -> None:
        with self._lock:
            if self._connection.closed:
                return
            try:
                with self._connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (self._lock_id,))
                self._connection.commit()
            finally:
                self._connection.close()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _seal_state(snapshot: dict[str, Any], state_key: bytes) -> dict[str, Any]:
    digest = hmac.new(state_key, _canonical_json(snapshot), hashlib.sha256).hexdigest()
    return {
        "schema": STATE_ENVELOPE_SCHEMA,
        "snapshot": snapshot,
        "hmac_sha256": digest,
    }


def _open_state(envelope: dict[str, Any], state_key: bytes) -> dict[str, Any]:
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema") != STATE_ENVELOPE_SCHEMA
    ):
        raise HolderServiceUnavailable("Holder state envelope is invalid")
    snapshot = envelope.get("snapshot")
    supplied = envelope.get("hmac_sha256")
    if not isinstance(snapshot, dict) or not isinstance(supplied, str):
        raise HolderServiceUnavailable("Holder state envelope is invalid")
    expected = hmac.new(
        state_key, _canonical_json(snapshot), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise HolderServiceUnavailable("Holder state authentication failed")
    return snapshot


class PersistentHolderRuntime:
    """Serialize holder mutations with authenticated durable state writes."""

    def __init__(
        self,
        holder: PresenceHolder,
        store: HolderStateStore,
        state_key: bytes,
    ) -> None:
        if not isinstance(state_key, bytes) or len(state_key) < 32:
            raise HolderServiceUnavailable("Holder state key must contain 32 bytes")
        self.holder = holder
        self._store = store
        self._state_key = state_key
        self._lock = RLock()
        self._available = True
        envelope, self._revision = store.load()
        if envelope is not None:
            try:
                holder.restore_state(_open_state(envelope, state_key))
            except (HolderPolicyError, HolderServiceUnavailable):
                self._available = False
                raise HolderServiceUnavailable(
                    "Stored holder state is invalid"
                ) from None

    @property
    def available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise HolderServiceUnavailable("Holder service is unavailable")

    def _mutate(self, operation: Callable[[], Any]) -> Any:
        with self._lock:
            self._require_available()
            result = operation()
            try:
                envelope = _seal_state(self.holder.export_state(), self._state_key)
                self._revision = self._store.save(envelope, self._revision)
            except Exception:
                self._available = False
                raise HolderServiceUnavailable(
                    "Holder state persistence failed"
                ) from None
            return result

    def authorize_session(
        self, *, issuer: str, session_id: str, presence: dict[str, Any]
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self.holder.authorize_session(
                issuer=issuer, session_id=session_id, presence=presence
            )
            return self.holder.status()

        return self._mutate(operation)

    def sign_submission(self, **values: str) -> str:
        return self._mutate(lambda: self.holder.sign_submission(**values))

    def commit_submission(
        self, *, session_id: str, presence: dict[str, Any]
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self.holder.commit_submission(session_id=session_id, presence=presence)
            return self.holder.status()

        return self._mutate(operation)

    def register_credential(self, *, issuer: str, attestation: dict[str, Any]) -> str:
        return self._mutate(
            lambda: self.holder.register_credential(
                issuer=issuer, attestation=attestation
            )
        )

    def sign_presentation(self, **values: str) -> str:
        return self._mutate(lambda: self.holder.sign_presentation(**values))

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._require_available()
            return {**self.holder.status(), "state_revision": self._revision}

    def health(self) -> bool:
        return self._available and self._store.health()

    def close(self) -> None:
        self._available = False
        self._store.close()


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise HolderServiceUnavailable(f"{name} is required")
    return value


def _required_key_version_environment(name: str) -> int:
    value = _required_environment(name)
    if re.fullmatch(r"[1-9][0-9]{0,9}", value) is None:
        raise HolderServiceUnavailable(f"{name} is invalid")
    version = int(value)
    if version > 2147483647:
        raise HolderServiceUnavailable(f"{name} is invalid")
    return version


def _read_bounded_file(path: str, name: str, maximum: int) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise HolderServiceUnavailable(f"{name} must be an absolute regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError:
        raise HolderServiceUnavailable(f"{name} could not be read") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_size < 1
            or metadata.st_size > maximum
        ):
            raise HolderServiceUnavailable(f"{name} permissions or size are invalid")
        data = os.read(descriptor, maximum + 1)
    finally:
        os.close(descriptor)
    if not data or len(data) > maximum:
        raise HolderServiceUnavailable(f"{name} is empty or oversized")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise HolderServiceUnavailable(f"{name} is not UTF-8") from None


@dataclass(frozen=True)
class HolderServiceSettings:
    holder_id: str
    vault_url: str
    vault_mount_path: str
    vault_key_name: str
    vault_key_version: int
    vault_public_key_file: str
    vault_token_file: str
    policy_file: str
    control_token_file: str
    state_hmac_key_file: str
    database_url: str

    @classmethod
    def from_environment(cls) -> "HolderServiceSettings":
        holder_id = _required_environment("METTLE_HOLDER_ID")
        if HOLDER_ID_PATTERN.fullmatch(holder_id) is None:
            raise HolderServiceUnavailable("METTLE_HOLDER_ID is invalid")
        return cls(
            holder_id=holder_id,
            vault_url=_required_environment("METTLE_HOLDER_VAULT_URL"),
            vault_mount_path=os.environ.get("METTLE_HOLDER_VAULT_MOUNT", "transit"),
            vault_key_name=os.environ.get("METTLE_HOLDER_VAULT_KEY", "mettle-holder"),
            vault_key_version=_required_key_version_environment(
                "METTLE_HOLDER_VAULT_KEY_VERSION"
            ),
            vault_public_key_file=_required_environment(
                "METTLE_HOLDER_VAULT_PUBLIC_KEY_FILE"
            ),
            vault_token_file=_required_environment("METTLE_HOLDER_VAULT_TOKEN_FILE"),
            policy_file=_required_environment("METTLE_HOLDER_POLICY_FILE"),
            control_token_file=_required_environment(
                "METTLE_HOLDER_CONTROL_TOKEN_FILE"
            ),
            state_hmac_key_file=_required_environment(
                "METTLE_HOLDER_STATE_HMAC_KEY_FILE"
            ),
            database_url=_required_environment("METTLE_HOLDER_DATABASE_URL"),
        )


def _load_policy(path: str) -> HolderPolicy:
    try:
        raw = json.loads(_read_bounded_file(path, "Holder policy file", 262144))
    except json.JSONDecodeError:
        raise HolderServiceUnavailable("Holder policy file is invalid JSON") from None
    if not isinstance(raw, dict):
        raise HolderServiceUnavailable("Holder policy must be an object")
    allowed_fields = {
        "issuer_public_keyrings",
        "allowed_audiences",
        "max_active_sessions",
        "max_actions_per_session",
        "max_presentations_per_credential",
        "max_presentation_ttl_seconds",
        "max_session_records",
        "max_credentials",
        "max_presentation_records",
    }
    if set(raw) - allowed_fields:
        raise HolderServiceUnavailable("Holder policy contains unsupported fields")
    keyrings = raw.get("issuer_public_keyrings")
    audiences = raw.get("allowed_audiences")
    if not isinstance(keyrings, dict) or not isinstance(audiences, list):
        raise HolderServiceUnavailable("Holder policy trust fields are invalid")

    def integer(name: str, default: int) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise HolderServiceUnavailable("Holder policy values are invalid")
        return value

    try:
        return HolderPolicy(
            issuer_public_keys={},
            issuer_public_keyrings=keyrings,
            allowed_audiences=frozenset(audiences),
            max_active_sessions=integer("max_active_sessions", 4),
            max_actions_per_session=integer("max_actions_per_session", 16),
            max_presentations_per_credential=integer(
                "max_presentations_per_credential", 32
            ),
            max_presentation_ttl_seconds=integer("max_presentation_ttl_seconds", 600),
            max_session_records=integer("max_session_records", 4096),
            max_credentials=integer("max_credentials", 4096),
            max_presentation_records=integer("max_presentation_records", 100000),
        )
    except (TypeError, ValueError):
        raise HolderServiceUnavailable("Holder policy values are invalid") from None


def build_runtime_from_environment() -> tuple[
    PersistentHolderRuntime, Callable[[], str]
]:
    settings = HolderServiceSettings.from_environment()
    vault_token_provider = FileSecretProvider(settings.vault_token_file)
    control_token_provider = FileSecretProvider(settings.control_token_file)
    state_secret = FileSecretProvider(settings.state_hmac_key_file)().encode("utf-8")
    public_key_pem = _read_bounded_file(
        settings.vault_public_key_file, "Vault public key file", 32768
    )
    signer = VaultTransitEd25519Signer(
        base_url=settings.vault_url,
        mount_path=settings.vault_mount_path,
        key_name=settings.vault_key_name,
        public_key_pem=public_key_pem,
        token_provider=vault_token_provider,
        key_version=settings.vault_key_version,
    )
    holder = PresenceHolder(signer, _load_policy(settings.policy_file))
    store = PostgresHolderStateStore(settings.database_url, settings.holder_id)
    try:
        runtime = PersistentHolderRuntime(holder, store, state_secret)
    except Exception:
        store.close()
        raise
    return runtime, control_token_provider


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthorizeSessionRequest(StrictModel):
    issuer: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1, max_length=256)
    presence: dict[str, Any]


class SignSubmissionRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=256)
    action: str = Field(min_length=1, max_length=134)
    nonce: str = Field(min_length=32, max_length=256)
    previous_transcript_hash: str = Field(min_length=71, max_length=71)
    payload_hash: str = Field(min_length=71, max_length=71)


class CommitSubmissionRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=256)
    presence: dict[str, Any]


class RegisterCredentialRequest(StrictModel):
    issuer: str = Field(min_length=1, max_length=512)
    attestation: dict[str, Any]


class SignPresentationRequest(StrictModel):
    challenge_id: str = Field(min_length=1, max_length=256)
    nonce: str = Field(min_length=32, max_length=256)
    audience: str = Field(min_length=1, max_length=256)
    credential_jti: str = Field(min_length=1, max_length=256)
    expires_at: str = Field(min_length=1, max_length=256)


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, maximum_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.maximum_bytes:
                    response = JSONResponse(
                        {"detail": "Request body is too large"}, status_code=413
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(
                    {"detail": "Content-Length is invalid"}, status_code=400
                )
                await response(scope, receive, send)
                return
        consumed = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.maximum_bytes:
                    raise HolderPolicyError("Request body is too large")
            return message

        try:
            await self.app(scope, limited_receive, send)
        except HolderPolicyError as exc:
            if str(exc) != "Request body is too large":
                raise
            response = JSONResponse(
                {"detail": "Request body is too large"}, status_code=413
            )
            await response(scope, receive, send)


def create_holder_service(
    *,
    runtime: PersistentHolderRuntime | None = None,
    control_token_provider: Callable[[], str] | None = None,
) -> FastAPI:
    owned_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime, control_token_provider
        if runtime is None or control_token_provider is None:
            runtime, control_token_provider = build_runtime_from_environment()
        application.state.runtime = runtime
        application.state.control_token_provider = control_token_provider
        try:
            yield
        finally:
            if owned_runtime and runtime is not None:
                runtime.close()

    service = FastAPI(
        title="METTLE Holder Service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    service.add_middleware(RequestBodyLimitMiddleware)

    @service.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @service.exception_handler(HolderPolicyError)
    async def holder_policy_error(
        _request: Request, exc: HolderPolicyError
    ) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @service.exception_handler(HolderServiceUnavailable)
    async def holder_unavailable(
        _request: Request, _exc: HolderServiceUnavailable
    ) -> JSONResponse:
        return JSONResponse(
            {"detail": "Holder service is unavailable"}, status_code=503
        )

    @service.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse({"detail": "Request payload is invalid"}, status_code=400)

    def current_runtime() -> PersistentHolderRuntime:
        value = service.state.runtime
        if not isinstance(value, PersistentHolderRuntime):
            raise HolderServiceUnavailable("Holder runtime is unavailable")
        return value

    def authorize(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        provider = service.state.control_token_provider
        try:
            expected = provider()
        except Exception:
            raise HolderServiceUnavailable(
                "Holder control token is unavailable"
            ) from None
        prefix = "Bearer "
        supplied = (
            authorization[len(prefix) :]
            if isinstance(authorization, str) and authorization.startswith(prefix)
            else ""
        )
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @service.get("/health")
    def health(
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> Any:
        if not holder_runtime.health():
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {
            "status": "healthy",
            "key_fingerprint": holder_runtime.holder.key_fingerprint,
        }

    @service.get("/v1/public-key", dependencies=[Depends(authorize)])
    def public_key(
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, str]:
        return {
            "algorithm": "Ed25519",
            "public_key_pem": holder_runtime.holder.public_key_pem,
            "key_fingerprint": holder_runtime.holder.key_fingerprint,
        }

    @service.get("/v1/status", dependencies=[Depends(authorize)])
    def status(
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, Any]:
        return holder_runtime.status()

    @service.post("/v1/sessions/authorize", dependencies=[Depends(authorize)])
    def authorize_session(
        body: AuthorizeSessionRequest,
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, Any]:
        return holder_runtime.authorize_session(**body.model_dump())

    @service.post("/v1/submissions/sign", dependencies=[Depends(authorize)])
    def sign_submission(
        body: SignSubmissionRequest,
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, str]:
        return {"signature": holder_runtime.sign_submission(**body.model_dump())}

    @service.post("/v1/submissions/commit", dependencies=[Depends(authorize)])
    def commit_submission(
        body: CommitSubmissionRequest,
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, Any]:
        return holder_runtime.commit_submission(**body.model_dump())

    @service.post("/v1/credentials/register", dependencies=[Depends(authorize)])
    def register_credential(
        body: RegisterCredentialRequest,
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, str]:
        return {
            "credential_jti": holder_runtime.register_credential(**body.model_dump())
        }

    @service.post("/v1/presentations/sign", dependencies=[Depends(authorize)])
    def sign_presentation(
        body: SignPresentationRequest,
        holder_runtime: PersistentHolderRuntime = Depends(current_runtime),
    ) -> dict[str, str]:
        return {"signature": holder_runtime.sign_presentation(**body.model_dump())}

    return service


app = create_holder_service()
