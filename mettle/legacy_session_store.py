"""Redis authority for the public legacy verification-session API.

The legacy endpoints predate the v2 Redis state machine.  This store keeps their
wire contract intact while making session progress safe across API workers.  A
short, token-owned Redis lock serializes each mutation; the session payload is
written as one value so readers see either the old or new state, never a partial
transition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mettle.models import MettleSession

logger = logging.getLogger(__name__)

LEGACY_SESSION_TTL_SECONDS = 1800
LEGACY_SESSION_LOCK_SECONDS = 30
LEGACY_SESSION_NAMESPACE = "mettle:legacy:session"

_LOCK_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class LegacySessionBusyError(RuntimeError):
    """Another worker currently owns the session mutation lock."""


class LegacySessionStateError(RuntimeError):
    """Redis rejected or returned malformed legacy-session state."""


@dataclass(frozen=True)
class LegacySessionRecord:
    """One authoritative legacy session and its active challenge clock."""

    session: MettleSession
    issued_at: float | None


def _key(session_id: str) -> str:
    return f"{LEGACY_SESSION_NAMESPACE}:{session_id}"


def _lock_key(session_id: str) -> str:
    return f"{_key(session_id)}:lock"


def _remaining_ttl(session: MettleSession) -> int:
    expires_at = session.started_at.timestamp() + LEGACY_SESSION_TTL_SECONDS
    return max(1, int(expires_at - time.time()))


def _serialize(record: LegacySessionRecord) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "session": record.session.model_dump(mode="json"),
            "issued_at": record.issued_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize(raw: str | bytes) -> LegacySessionRecord:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        issued_at = payload.get("issued_at")
        if issued_at is not None:
            issued_at = float(issued_at)
        return LegacySessionRecord(
            session=MettleSession.model_validate(payload["session"]),
            issued_at=issued_at,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise LegacySessionStateError("Malformed legacy session state") from exc


class LegacySessionStore:
    """Persist and serialize legacy session mutations through Redis."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client

    async def create(self, record: LegacySessionRecord) -> None:
        """Create a session without replacing an existing identifier."""
        created = await self.redis.set(
            _key(record.session.session_id),
            _serialize(record),
            ex=_remaining_ttl(record.session),
            nx=True,
        )
        if not created:
            raise LegacySessionStateError("Legacy session identifier collision")

    async def load(self, session_id: str) -> LegacySessionRecord | None:
        """Load current authoritative state, or ``None`` after expiry."""
        raw = await self.redis.get(_key(session_id))
        return None if raw is None else _deserialize(raw)

    async def save(self, record: LegacySessionRecord) -> None:
        """Replace a session only while the authoritative key still exists."""
        saved = await self.redis.set(
            _key(record.session.session_id),
            _serialize(record),
            ex=_remaining_ttl(record.session),
            xx=True,
        )
        if not saved:
            raise LegacySessionStateError("Legacy session expired during update")

    async def delete(self, session_id: str) -> None:
        """Delete a session whose creation could not be fully committed."""
        await self.redis.delete(_key(session_id))

    @asynccontextmanager
    async def mutation(self, session_id: str) -> AsyncIterator[None]:
        """Hold the distributed, token-owned mutation lock for one transition."""
        token = secrets.token_urlsafe(24)
        acquired = await self.redis.set(
            _lock_key(session_id),
            token,
            ex=LEGACY_SESSION_LOCK_SECONDS,
            nx=True,
        )
        if not acquired:
            raise LegacySessionBusyError("Legacy session update already in progress")
        try:
            yield
        finally:
            try:
                await asyncio.shield(
                    self.redis.eval(
                        _LOCK_RELEASE_SCRIPT,
                        1,
                        _lock_key(session_id),
                        token,
                    )
                )
            except Exception as exc:
                # The token check and short TTL make expiry the safe fallback.
                # A release outage must not conceal an already committed answer.
                logger.error(
                    "legacy session lock release failed: %s",
                    type(exc).__name__,
                )
