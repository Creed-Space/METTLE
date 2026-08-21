"""Tests targeting uncovered lines in mettle/router.py.

Coverage gaps addressed:
- Line 53: get_session_manager returns SessionManager when Redis is present
- Lines 141-143: create_session generic Exception handler
- Lines 239-241: verify_single_shot generic Exception handler
- Lines 286-288: submit_round_answer generic Exception handler
- Lines 360-373: VCP attestation branch in get_session_result (include_vcp=true)
- Lines 393-398: .well-known/vcp-keys when mettle.signing ImportError

Uses the same FakeRedis + dependency-override pattern as test_api_coverage.py.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mettle.api_models import SessionStatus
from mettle.session_manager import SessionManager


# ---------------------------------------------------------------------------
# FakeRedis (same pattern as test_api_coverage.py / test_mettle_api.py)
# ---------------------------------------------------------------------------


class FakeRedis:
    """In-memory async Redis mock for testing."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}
        self._ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = ttl

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        if nx and key in self._store:
            return False
        self._store[key] = value
        if ex is not None:
            self._ttls[key] = ex
        return True

    async def delete(self, key: str) -> int:
        return int(self._store.pop(key, None) is not None)

    async def eval(
        self,
        _script: str,
        numkeys: int,
        *args: Any,
    ) -> int:
        if numkeys == 1 and len(args) == 2:
            key, expected_value = args
            if self._store.get(key) != expected_value:
                return 0
            if "DEL" in _script:
                return await self.delete(key)
            return 1

        if numkeys == 1 and len(args) == 3:
            key, expected_value, ttl = args
            if self._store.get(key) != expected_value:
                return 0
            self._ttls[key] = int(ttl)
            return 1

        if numkeys == 2 and len(args) == 5:
            session_key, lock_key, payload, ttl, lock_token = args
            if self._store.get(lock_key) != lock_token:
                return 0
            self._store[session_key] = payload
            self._ttls[session_key] = int(ttl)
            return 1

        if numkeys == 5 and len(args) == 9:
            (
                session_key,
                active_key,
                answers_key,
                lock_key,
                reservation_key,
                payload,
                ttl,
                session_id,
                lock_token,
            ) = args
            if self._store.get(lock_key) != lock_token:
                return 0
            self._store[session_key] = payload
            self._ttls[session_key] = int(ttl)
            self._sets.get(active_key, set()).discard(session_id)
            self._store.pop(answers_key, None)
            self._store.pop(reservation_key, None)
            return 1

        if numkeys == 3 and len(args) == 9:
            (
                active_key,
                hourly_key,
                reservation_key,
                session_id,
                max_active,
                max_hourly,
                active_ttl,
                hourly_ttl,
                _session_prefix,
            ) = args
            if len(self._sets.get(active_key, set())) >= int(max_active):
                return -1
            if int(self._store.get(hourly_key, "0")) >= int(max_hourly):
                return -2
            self._sets.setdefault(active_key, set()).add(session_id)
            self._ttls[active_key] = int(active_ttl)
            self._store[hourly_key] = str(int(self._store.get(hourly_key, "0")) + 1)
            self._ttls[hourly_key] = int(hourly_ttl)
            self._store[reservation_key] = "1"
            self._ttls[reservation_key] = int(active_ttl)
            return 1

        if numkeys == 3 and len(args) == 4:
            active_key, hourly_key, reservation_key, session_id = args
            self._sets.get(active_key, set()).discard(session_id)
            self._store.pop(reservation_key, None)
            hourly = int(self._store.get(hourly_key, "0"))
            self._store[hourly_key] = str(max(0, hourly - 1))
            return 1

        raise AssertionError("Unexpected Lua script in FakeRedis")

    async def sadd(self, key: str, *values: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].update(values)
        return len(values)

    async def srem(self, key: str, *values: str) -> int:
        if key not in self._sets:
            return 0
        before = len(self._sets[key])
        self._sets[key] -= set(values)
        return before - len(self._sets[key])

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, "0"))
        val += 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int) -> None:
        self._ttls[key] = ttl

    def pipeline(self) -> FakeRedisPipeline:
        return FakeRedisPipeline(self)


class FakeRedisPipeline:
    """Accumulates commands then executes them."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...]]] = []

    def setex(self, key: str, ttl: int, value: str) -> FakeRedisPipeline:
        self._commands.append(("setex", (key, ttl, value)))
        return self

    def sadd(self, key: str, *values: str) -> FakeRedisPipeline:
        self._commands.append(("sadd", (key, *values)))
        return self

    def srem(self, key: str, *values: str) -> FakeRedisPipeline:
        self._commands.append(("srem", (key, *values)))
        return self

    def incr(self, key: str) -> FakeRedisPipeline:
        self._commands.append(("incr", (key,)))
        return self

    def expire(self, key: str, ttl: int) -> FakeRedisPipeline:
        self._commands.append(("expire", (key, ttl)))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args in self._commands:
            fn = getattr(self._redis, cmd)
            result = await fn(*args)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_user(user_id: str = "test-user-123") -> MagicMock:
    user = MagicMock()
    user.user_id = user_id
    return user


def _operator_commitment(entity_id: str, nonce: str = "c" * 64) -> dict[str, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    issued_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    commitment = {
        "operator_pseudonym": "operator-1",
        "operator_public_key": public_key,
        "contact_method": "email_hash",
        "contact_hash": "a" * 64,
        "issued_at": issued_at,
        "nonce": nonce,
    }
    message = json.dumps(
        {
            **commitment,
            "entity_id": entity_id,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    commitment["signed_commitment"] = base64.b64encode(
        private_key.sign(message)
    ).decode()
    return commitment


def _build_app(user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    """Build a test FastAPI app with dependency overrides."""
    from mettle.auth import require_authenticated_user
    from mettle.router import get_session_manager, router

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[require_authenticated_user] = lambda: user

    async def mock_get_manager() -> SessionManager:
        return SessionManager(fake_redis)

    app.dependency_overrides[get_session_manager] = mock_get_manager
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def mock_user() -> MagicMock:
    return _make_mock_user()


@pytest.fixture
def app(mock_user: MagicMock, fake_redis: FakeRedis) -> FastAPI:
    return _build_app(mock_user, fake_redis)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# get_session_manager: Redis=None -> 503 (line 49-52) and success path (line 53)
# ---------------------------------------------------------------------------


class TestGetSessionManagerDependency:
    """Covers lines 47-53 of router.py."""

    @pytest.mark.asyncio
    async def test_redis_none_raises_503(self) -> None:
        """Line 49-52: when request.app.state.redis is None, raise 503."""
        from fastapi import HTTPException
        from mettle.router import get_session_manager

        mock_request = MagicMock()
        mock_request.app.state.redis = None

        with pytest.raises(HTTPException) as exc_info:
            await get_session_manager(mock_request)
        assert exc_info.value.status_code == 503
        assert "Redis" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_redis_missing_attr_raises_503(self) -> None:
        """Line 47-52: when request.app.state has no redis attr -> None -> 503."""
        from fastapi import HTTPException
        from mettle.router import get_session_manager

        mock_request = MagicMock()
        # getattr(request.app.state, "redis", None) returns None when attr missing
        del mock_request.app.state.redis

        with pytest.raises(HTTPException) as exc_info:
            await get_session_manager(mock_request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_redis_present_returns_session_manager(self) -> None:
        """Line 53: returns SessionManager(redis) when Redis is available."""
        from mettle.router import get_session_manager

        fake_redis_obj = FakeRedis()
        mock_request = MagicMock()
        mock_request.app.state.redis = fake_redis_obj

        result = await get_session_manager(mock_request)
        assert isinstance(result, SessionManager)
        assert result.redis is fake_redis_obj


# ---------------------------------------------------------------------------
# create_session: generic Exception -> 500 (lines 141-143)
# ---------------------------------------------------------------------------


class TestCreateSessionGenericException:
    """Covers lines 141-143 of router.py."""

    def test_generic_exception_returns_500(self, fake_redis: FakeRedis) -> None:
        """When SessionManager.create_session raises a non-ValueError Exception,
        the endpoint catches it and returns 500."""
        user = _make_mock_user()
        from mettle.auth import require_authenticated_user
        from mettle.router import get_session_manager, router

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_authenticated_user] = lambda: user

        # Create a SessionManager that raises a generic exception on create_session
        async def mock_get_manager() -> SessionManager:
            mgr = SessionManager(fake_redis)

            # Monkey-patch create_session to raise a generic exception
            async def exploding(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Redis connection lost")

            mgr.create_session = exploding  # type: ignore[assignment]
            return mgr

        app.dependency_overrides[get_session_manager] = mock_get_manager
        test_client = TestClient(app)

        resp = test_client.post(
            "/api/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        assert resp.status_code == 500
        assert "Failed to create session" in resp.json()["detail"]

    def test_redis_exception_returns_503(self, fake_redis: FakeRedis) -> None:
        from mettle.auth import require_authenticated_user
        from mettle.router import get_session_manager, router
        from redis.exceptions import ConnectionError as RedisConnectionError

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_authenticated_user] = _make_mock_user

        async def mock_get_manager() -> SessionManager:
            manager = SessionManager(fake_redis)

            async def unavailable(*args: Any, **kwargs: Any) -> Any:
                raise RedisConnectionError("session store unavailable")

            manager.create_session = unavailable  # type: ignore[assignment]
            return manager

        app.dependency_overrides[get_session_manager] = mock_get_manager
        response = TestClient(app).post(
            "/api/mettle/sessions", json={"suites": ["adversarial"]}
        )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "METTLE session storage temporarily unavailable"
        )


class TestOperatorCommitmentNonceReservation:
    """Operator commitments are fresh, single-use, and rollback-safe."""

    def test_replay_is_rejected_after_atomic_nonce_reservation(
        self, fake_redis: FakeRedis
    ) -> None:
        user = _make_mock_user()
        test_client = TestClient(_build_app(user, fake_redis))
        commitment = _operator_commitment("agent-1")
        body = {
            "suites": ["adversarial"],
            "entity_id": "agent-1",
            "operator_commitment": commitment,
        }

        first = test_client.post("/api/mettle/sessions", json=body)
        replay = test_client.post("/api/mettle/sessions", json=body)

        assert first.status_code == 201
        assert replay.status_code == 400
        assert "already been used" in replay.json()["detail"]
        nonce_key = "mettle:operator-commitment:nonce:" + commitment["nonce"]
        assert nonce_key in fake_redis._store
        assert fake_redis._ttls[nonce_key] >= 330

    def test_failed_session_creation_releases_only_its_nonce(
        self, fake_redis: FakeRedis
    ) -> None:
        user = _make_mock_user()
        from mettle.auth import require_authenticated_user
        from mettle.router import get_session_manager, router

        async def failing_manager() -> SessionManager:
            manager = SessionManager(fake_redis)

            async def explode(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("challenge generation failed")

            manager.create_session = explode  # type: ignore[assignment]
            return manager

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[require_authenticated_user] = lambda: user
        test_app.dependency_overrides[get_session_manager] = failing_manager
        commitment = _operator_commitment("agent-1", nonce="d" * 64)

        response = TestClient(test_app).post(
            "/api/mettle/sessions",
            json={
                "suites": ["adversarial"],
                "entity_id": "agent-1",
                "operator_commitment": commitment,
            },
        )

        assert response.status_code == 500
        nonce_key = "mettle:operator-commitment:nonce:" + commitment["nonce"]
        assert nonce_key not in fake_redis._store


# ---------------------------------------------------------------------------
# verify_single_shot: generic Exception -> 500 (lines 239-241)
# ---------------------------------------------------------------------------


class TestVerifySingleShotGenericException:
    """Covers lines 239-241 of router.py."""

    def test_generic_exception_returns_500(self, fake_redis: FakeRedis) -> None:
        """When mgr.verify_single_shot raises a non-ValueError/non-HTTPException,
        the endpoint returns 500."""
        user = _make_mock_user()
        from mettle.auth import require_authenticated_user
        from mettle.router import get_session_manager, router

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_authenticated_user] = lambda: user

        async def mock_get_manager() -> SessionManager:
            mgr = SessionManager(fake_redis)

            # Patch verify_single_shot to explode after session/ownership checks pass
            async def exploding(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Unexpected evaluator crash")

            mgr.verify_single_shot = exploding  # type: ignore[assignment]
            return mgr

        app.dependency_overrides[get_session_manager] = mock_get_manager
        test_client = TestClient(app)

        # Create a real session first (using the real manager for initial storage)
        create_app = _build_app(user, fake_redis)
        create_client = TestClient(create_app)
        create_resp = create_client.post(
            "/api/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Now try to verify using the patched manager that will explode
        resp = test_client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={"suite": "adversarial", "answers": {}},
        )
        assert resp.status_code == 500
        assert "Verification failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# submit_round_answer: generic Exception -> 500 (lines 286-288)
# ---------------------------------------------------------------------------


class TestSubmitRoundGenericException:
    """Covers lines 286-288 of router.py."""

    def test_generic_exception_returns_500(self, fake_redis: FakeRedis) -> None:
        """When mgr.submit_round_answer raises a generic Exception,
        the endpoint returns 500."""
        user = _make_mock_user()
        from mettle.auth import require_authenticated_user
        from mettle.router import get_session_manager, router

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_authenticated_user] = lambda: user

        async def mock_get_manager() -> SessionManager:
            mgr = SessionManager(fake_redis)

            async def exploding(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Novel reasoning engine crashed")

            mgr.submit_round_answer = exploding  # type: ignore[assignment]
            return mgr

        app.dependency_overrides[get_session_manager] = mock_get_manager
        test_client = TestClient(app)

        # Create a real session with novel-reasoning
        create_app = _build_app(user, fake_redis)
        create_client = TestClient(create_app)
        create_resp = create_client.post(
            "/api/mettle/sessions",
            json={"suites": ["novel-reasoning"], "difficulty": "easy"},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Try to submit round with the patched manager
        resp = test_client.post(
            f"/api/mettle/sessions/{session_id}/rounds/1/answer",
            json={"answers": {}},
        )
        assert resp.status_code == 500
        assert "Round submission failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# get_session_result with include_vcp=true (lines 360-373)
# ---------------------------------------------------------------------------


class TestGetSessionResultVCPAttestation:
    """Covers lines 359-382 of router.py - the VCP attestation branch."""

    def _complete_single_suite_session(
        self, fake_redis: FakeRedis, user: MagicMock
    ) -> str:
        """Create and complete a single-suite session, return session_id."""
        create_app = _build_app(user, fake_redis)
        create_client = TestClient(create_app)

        # Create session with a single suite
        create_resp = create_client.post(
            "/api/mettle/sessions",
            json={"suites": ["adversarial"]},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]
        challenges = create_resp.json()["challenges"]

        # Verify the suite to complete the session
        verify_resp = create_client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={
                "suite": "adversarial",
                "answers": challenges.get("adversarial", {}),
            },
        )
        assert verify_resp.status_code == 200

        # Verify session is completed
        key = f"mettle:session:{session_id}"
        stored = fake_redis._store.get(key)
        assert stored is not None
        session_data = json.loads(stored)
        assert session_data["status"] == SessionStatus.COMPLETED.value

        return session_id

    def test_include_vcp_true_returns_attestation(self, fake_redis: FakeRedis) -> None:
        """Lines 359-380: include_vcp=true triggers VCP attestation building."""
        user = _make_mock_user()
        session_id = self._complete_single_suite_session(fake_redis, user)

        # Fetch result with include_vcp=true
        result_app = _build_app(user, fake_redis)
        result_client = TestClient(result_app)

        resp = result_client.get(
            f"/api/mettle/sessions/{session_id}/result?include_vcp=true"
        )
        assert resp.status_code == 200
        data = resp.json()

        # VCP attestation should be present
        assert data["vcp_attestation"] is not None
        att = data["vcp_attestation"]
        assert att["auditor"] == "mettle.creed.space"
        assert att["attestation_type"] == "mettle-evidence-receipt"
        assert att["signature"] is None
        assert att["metadata"]["credential_eligible"] is False
        assert data["credential_eligible"] is False
        assert data["tier"] == "none"
        assert "metadata" in att
        assert "tier" in att["metadata"]
        assert "content_hash" in att

    def test_include_vcp_false_returns_no_attestation(
        self, fake_redis: FakeRedis
    ) -> None:
        """Baseline: include_vcp=false (default) returns null attestation."""
        user = _make_mock_user()
        session_id = self._complete_single_suite_session(fake_redis, user)

        result_app = _build_app(user, fake_redis)
        result_client = TestClient(result_app)

        resp = result_client.get(f"/api/mettle/sessions/{session_id}/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vcp_attestation"] is None

    def test_signing_availability_cannot_promote_evidence(
        self, fake_redis: FakeRedis
    ) -> None:
        """A configured signer must not sign behavioral evidence."""
        user = _make_mock_user()
        session_id = self._complete_single_suite_session(fake_redis, user)

        result_app = _build_app(user, fake_redis)
        result_client = TestClient(result_app)

        # Mock the signing module so is_available returns True and
        # sign_attestation returns a fake signature
        with (
            patch("mettle.signing.is_available", return_value=True),
            patch("mettle.signing.sign_attestation", return_value="fake_sig_base64"),
        ):
            resp = result_client.get(
                f"/api/mettle/sessions/{session_id}/result?include_vcp=true"
            )

        assert resp.status_code == 200
        data = resp.json()
        att = data["vcp_attestation"]
        assert att is not None
        assert att["signature"] is None
        assert att["metadata"]["tier"] == "none"

    def test_include_vcp_signing_import_error(self, fake_redis: FakeRedis) -> None:
        """Lines 369-370: when mettle.signing raises ImportError,
        sign_fn stays None and attestation has no signature."""
        user = _make_mock_user()
        session_id = self._complete_single_suite_session(fake_redis, user)

        result_app = _build_app(user, fake_redis)
        result_client = TestClient(result_app)

        # Make the import of mettle.signing fail inside the endpoint
        import builtins

        original_import = builtins.__import__

        def import_blocker(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mettle.signing":
                raise ImportError("No module named 'cryptography'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            resp = result_client.get(
                f"/api/mettle/sessions/{session_id}/result?include_vcp=true"
            )

        assert resp.status_code == 200
        data = resp.json()
        att = data["vcp_attestation"]
        assert att is not None
        # Signature should be None because signing was unavailable
        assert att["signature"] is None


# ---------------------------------------------------------------------------
# .well-known/vcp-keys: ImportError branch (lines 393-404)
# ---------------------------------------------------------------------------


class TestWellKnownVCPKeys:
    """Covers lines 387-404 of router.py."""

    def test_vcp_keys_normal(self, client: TestClient) -> None:
        """Lines 393-396: when mettle.signing is available, returns key info."""
        with patch(
            "mettle.signing.get_public_key_info",
            return_value={
                "key_id": "mettle-vcp-v1",
                "algorithm": "Ed25519",
                "public_key_pem": "-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----",
                "available": True,
            },
        ):
            resp = client.get("/api/mettle/.well-known/vcp-keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["key_id"] == "mettle-vcp-v1"

    def test_vcp_keys_import_error(self, client: TestClient) -> None:
        """Lines 397-404: when mettle.signing cannot be imported, returns
        fallback dict with available=False."""
        import builtins

        original_import = builtins.__import__

        def import_blocker(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mettle.signing":
                raise ImportError("No module named 'cryptography'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            resp = client.get("/api/mettle/.well-known/vcp-keys")

        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["public_key_pem"] is None
        assert "cryptography" in data["error"]
        assert data["key_id"] == "mettle-vcp-v1"
        assert data["algorithm"] == "Ed25519"
