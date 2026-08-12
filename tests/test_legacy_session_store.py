"""Redis legacy-session authority and concurrency tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import main
from mettle.legacy_session_store import (
    LegacySessionBusyError,
    LegacySessionRecord,
    LegacySessionStateError,
    LegacySessionStore,
)
from mettle.models import Challenge, ChallengeType, Difficulty, MettleSession


class FakeRedis:
    """Small async Redis double covering the store's atomic primitives."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        del ex
        if nx and key in self.values:
            return False
        if xx and key not in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        token: str,
    ) -> int:
        del script, numkeys
        if self.values.get(key) != token:
            return 0
        return await self.delete(key)


def _record(session_id: str = "ses_" + "a" * 24) -> LegacySessionRecord:
    challenge = Challenge(
        id="mtl_" + "b" * 24,
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate 1 + 1",
        data={"expected_answer": 2},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=5000,
    )
    return LegacySessionRecord(
        session=MettleSession(
            session_id=session_id,
            entity_id="agent-test",
            difficulty=Difficulty.BASIC,
            challenges=[challenge],
            access_token_hash="c" * 64,
        ),
        issued_at=1234.5,
    )


@pytest.mark.asyncio
async def test_create_load_save_and_delete_round_trip() -> None:
    redis = FakeRedis()
    store = LegacySessionStore(redis)
    record = _record()

    await store.create(record)
    loaded = await store.load(record.session.session_id)
    assert loaded is not None
    assert loaded.session == record.session
    assert loaded.issued_at == 1234.5

    loaded.session.completed = True
    updated = LegacySessionRecord(session=loaded.session, issued_at=None)
    await store.save(updated)
    assert (await store.load(record.session.session_id)) == updated

    await store.delete(record.session.session_id)
    assert await store.load(record.session.session_id) is None


@pytest.mark.asyncio
async def test_create_does_not_replace_and_save_does_not_resurrect() -> None:
    redis = FakeRedis()
    store = LegacySessionStore(redis)
    record = _record()

    await store.create(record)
    with pytest.raises(LegacySessionStateError):
        await store.create(record)

    await store.delete(record.session.session_id)
    with pytest.raises(LegacySessionStateError):
        await store.save(record)


@pytest.mark.asyncio
async def test_mutation_lock_is_exclusive_and_released() -> None:
    redis = FakeRedis()
    store = LegacySessionStore(redis)
    session_id = _record().session.session_id

    async with store.mutation(session_id):
        with pytest.raises(LegacySessionBusyError):
            async with store.mutation(session_id):
                pytest.fail("contending worker acquired the same session")

    async with store.mutation(session_id):
        pass


@pytest.mark.asyncio
async def test_cancelled_mutation_releases_its_token_owned_lock() -> None:
    redis = FakeRedis()
    store = LegacySessionStore(redis)
    session_id = _record().session.session_id
    entered = asyncio.Event()

    async def hold_lock() -> None:
        async with store.mutation(session_id):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_lock())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with store.mutation(session_id):
        pass


@pytest.mark.asyncio
async def test_malformed_or_unknown_schema_fails_closed() -> None:
    redis = FakeRedis()
    store = LegacySessionStore(redis)
    session_id = _record().session.session_id
    key = f"mettle:legacy:session:{session_id}"

    redis.values[key] = "not-json"
    with pytest.raises(LegacySessionStateError):
        await store.load(session_id)

    redis.values[key] = '{"schema_version":99,"session":{}}'
    with pytest.raises(LegacySessionStateError):
        await store.load(session_id)


def test_fake_redis_matches_expected_async_surface() -> None:
    """Keep the double honest when store operations change."""
    required: tuple[str, ...] = ("set", "get", "delete", "eval")
    assert all(callable(getattr(FakeRedis(), name, None)) for name in required)
    assert isinstance(FakeRedis().values, dict)


def test_legacy_flow_crosses_independent_clients_through_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No process-local session or challenge entry is needed between workers."""
    redis = FakeRedis()
    monkeypatch.setattr(main.settings, "redis_url", "redis://test/0")
    main.app.state.redis = redis
    worker_a = TestClient(main.app)
    worker_b = TestClient(main.app)

    started = worker_a.post("/api/session/start", json={}).json()
    session_id = started["session_id"]
    token = started["session_token"]
    challenge = started["current_challenge"]
    headers = {"X-Session-Token": token}

    assert session_id not in main.sessions
    for worker in (worker_b, worker_a, worker_b):
        answered = worker.post(
            "/api/session/answer",
            json={
                "session_id": session_id,
                "challenge_id": challenge["id"],
                "answer": "bounded test response",
            },
            headers=headers,
        )
        assert answered.status_code == 200
        payload = answered.json()
        challenge = payload.get("next_challenge")
        if challenge is None:
            break

    result = worker_a.get(
        f"/api/session/{session_id}/result",
        headers=headers,
    )
    assert result.status_code == 200
    assert result.json()["total"] == 3
    assert session_id not in main.sessions
