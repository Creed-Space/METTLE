"""Security regression tests for the METTLE Presence Protocol."""

from __future__ import annotations

import base64
import copy
import json
import sys
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mettle.signing as issuer_signing
import mettle.session_manager as session_manager
from mettle.auth import AuthenticatedUser, require_authenticated_user
from mettle.continuity import (
    CONTINUITY_ANSWER_KEY,
    CONTINUITY_CHALLENGE_KEY,
    CONTINUITY_PROTOCOL,
    solve_continuity_challenge,
)
from mettle.presence import (
    answer_hash,
    presentation_signing_bytes,
    submission_signing_bytes,
)
from mettle.router import router


TEST_USER = "presence-verifier"
BRONZE_SUITES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
]


class FakeRedis:
    """Minimal Redis surface used by the Presence Protocol integration tests."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._sets: dict[str, set[str]] = {}

    async def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def setex(self, key: str, _ttl: int, value: Any) -> None:
        self._data[key] = value if isinstance(value, bytes) else str(value).encode()

    async def delete(self, key: str) -> int:
        return 1 if self._data.pop(key, None) is not None else 0

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def sadd(self, key: str, *members: str) -> int:
        self._sets.setdefault(key, set()).update(members)
        return len(members)

    async def srem(self, key: str, *members: str) -> int:
        current = self._sets.setdefault(key, set())
        before = len(current)
        current.difference_update(members)
        return before - len(current)

    async def incr(self, key: str) -> int:
        value = int(self._data.get(key, b"0")) + 1
        self._data[key] = str(value).encode()
        return value

    async def expire(self, _key: str, _ttl: int) -> None:
        return None

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str):
        if name not in {"setex", "sadd", "srem", "incr", "expire"}:
            raise AttributeError(name)

        def queue(*args: Any) -> FakePipeline:
            self.operations.append((name, args))
            return self

        return queue

    async def execute(self) -> list[Any]:
        results = []
        for name, args in self.operations:
            results.append(await getattr(self.redis, name)(*args))
        return results


def _keypair() -> tuple[Ed25519PrivateKey, str]:
    private_key = Ed25519PrivateKey.generate()
    public_pem = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    return private_key, public_pem


def _sign(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


@pytest.fixture(autouse=True)
def signing_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("METTLE_DEV_MODE", "true")
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False
    assert issuer_signing.init_signing() is True
    yield
    issuer_signing._private_key = None
    issuer_signing._public_key = None
    issuer_signing._initialized = False


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    fake_redis = FakeRedis()
    app = FastAPI()
    app.include_router(router)
    app.state.redis = fake_redis
    app.state.credential_revocation_checker = lambda _jti: False

    async def auth() -> AuthenticatedUser:
        return AuthenticatedUser(user_id=TEST_USER)

    app.dependency_overrides[require_authenticated_user] = auth

    client_stub = {"suite": "stub", "challenges": {"q1": {}}}
    server_stub = {"q1": {"expected": 42}}
    mock_engine = MagicMock()
    mock_engine.NovelReasoningChallenges.DIFFICULTY_PARAMS = {
        "easy": {"time_budget_s": 45},
        "standard": {"time_budget_s": 30},
        "hard": {"time_budget_s": 20},
    }
    mock_engine.IterationCurveAnalyzer.analyze_curve.return_value = {
        "overall": 0.9,
        "signature": "AI",
    }
    saved_engine = sys.modules.get("scripts.engine")
    sys.modules["scripts.engine"] = mock_engine
    generators = [
        "generate_adversarial",
        "generate_native",
        "generate_self_reference",
        "generate_social",
        "generate_inverse_turing",
    ]
    patches = [
        patch(
            f"mettle.session_manager.ChallengeAdapter.{name}",
            return_value=(client_stub, server_stub),
        )
        for name in generators
    ]
    patches.append(
        patch(
            "mettle.session_manager.ChallengeAdapter.evaluate_single_shot",
            return_value={"passed": True, "score": 1.0, "details": {}},
        )
    )
    patches.append(
        patch(
            "mettle.session_manager.ChallengeAdapter.generate_novel_reasoning",
            return_value=(
                {"round": 1, "challenges": {"seq": {}}},
                {
                    "time_budget_s": 30,
                    "num_rounds": 1,
                    "pass_threshold": 0.65,
                    "challenges": {"seq": {}},
                },
            ),
        )
    )
    patches.append(
        patch(
            "mettle.session_manager.ChallengeAdapter.evaluate_novel_round",
            return_value={"accuracy": 0.9, "errors": []},
        )
    )
    for active_patch in patches:
        active_patch.start()

    yield TestClient(app)

    for active_patch in patches:
        active_patch.stop()
    if saved_engine is None:
        sys.modules.pop("scripts.engine", None)
    else:
        sys.modules["scripts.engine"] = saved_engine
    app.dependency_overrides.clear()


def _answers_with_continuity(
    issued: dict[str, Any],
    suite: str,
    answers: dict[str, Any],
) -> dict[str, Any]:
    challenge = issued[suite][CONTINUITY_CHALLENGE_KEY]
    return {
        **answers,
        CONTINUITY_ANSWER_KEY: {
            "challenge_id": challenge["challenge_id"],
            "computed": solve_continuity_challenge(challenge),
        },
    }


def _complete_presence_session(
    client: TestClient,
    private_key: Ed25519PrivateKey,
    public_key_pem: str,
) -> dict[str, Any]:
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": BRONZE_SUITES,
            "difficulty": "standard",
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    )
    assert created.status_code == 201, created.text
    creation = created.json()
    session_id = creation["session_id"]
    state = creation["presence"]
    issued = creation["challenges"]
    assert list(creation["challenges"]) == [BRONZE_SUITES[0]]
    assert state["action"] == f"suite:{BRONZE_SUITES[0]}"

    for index, suite in enumerate(BRONZE_SUITES):
        answers = _answers_with_continuity(issued, suite, {"q1": 42})
        message = submission_signing_bytes(
            session_id=session_id,
            action=f"suite:{suite}",
            nonce=state["nonce"],
            previous_transcript_hash=state["transcript_hash"],
            payload_hash=answer_hash(answers),
        )
        response = client.post(
            f"/api/mettle/sessions/{session_id}/verify",
            json={
                "suite": suite,
                "answers": answers,
                "presence_proof": {
                    "nonce": state["nonce"],
                    "previous_transcript_hash": state["transcript_hash"],
                    "signature": _sign(private_key, message),
                },
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        state = body["presence"]
        if index + 1 < len(BRONZE_SUITES):
            next_suite = BRONZE_SUITES[index + 1]
            assert list(body["next_challenge"]) == [next_suite]
            assert state["action"] == f"suite:{next_suite}"
            issued = body["next_challenge"]
        else:
            assert body["next_challenge"] is None
            assert state["action"] is None

    result = client.get(f"/api/mettle/sessions/{session_id}/result?include_vcp=true")
    assert result.status_code == 200, result.text
    return result.json()["vcp_attestation"]


def test_presence_session_rejects_unsigned_and_wrong_key_submissions(
    client: TestClient,
) -> None:
    private_key, public_key_pem = _keypair()
    wrong_key, _ = _keypair()
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["adversarial"],
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    )
    creation = created.json()
    state = creation["presence"]
    session_id = creation["session_id"]
    answers = _answers_with_continuity(
        creation["challenges"], "adversarial", {"q1": 42}
    )

    unsigned = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={"suite": "adversarial", "answers": answers},
    )
    assert unsigned.status_code == 400
    assert "presence proof is required" in unsigned.json()["detail"].lower()

    message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce=state["nonce"],
        previous_transcript_hash=state["transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    copied = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={
            "suite": "adversarial",
            "answers": answers,
            "presence_proof": {
                "nonce": state["nonce"],
                "previous_transcript_hash": state["transcript_hash"],
                "signature": _sign(wrong_key, message),
            },
        },
    )
    assert copied.status_code == 400
    assert "signature" in copied.json()["detail"].lower()

    valid = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={
            "suite": "adversarial",
            "answers": answers,
            "presence_proof": {
                "nonce": state["nonce"],
                "previous_transcript_hash": state["transcript_hash"],
                "signature": _sign(private_key, message),
            },
        },
    )
    assert valid.status_code == 200, valid.text


def test_key_bound_credential_requires_fresh_holder_proof_and_rejects_replay(
    client: TestClient,
) -> None:
    holder_key, public_key_pem = _keypair()
    copied_key, _ = _keypair()
    attestation = _complete_presence_session(client, holder_key, public_key_pem)
    metadata = attestation["metadata"]
    assert attestation["attestation_type"] == "mettle-presence-credential"
    assert metadata["proof_of_possession"]["key_fingerprint"].startswith("sha256:")
    timing = metadata["proof_of_possession"]["server_timing"]
    continuity = metadata["proof_of_possession"]["continuity"]
    assert continuity["protocol"] == CONTINUITY_PROTOCOL
    assert continuity["challenge_count"] == len(BRONZE_SUITES)
    assert continuity["transcript_bound"] is True
    assert timing["total_elapsed_ms"] >= 0
    assert len(timing["submissions"]) == len(BRONZE_SUITES)
    assert [item["sequence"] for item in timing["submissions"]] == [1, 2, 3, 4, 5]

    challenge_response = client.post(
        "/api/mettle/presentation-challenges",
        json={
            "credential_jti": metadata["jti"],
            "audience": "service.example",
        },
    )
    assert challenge_response.status_code == 201
    challenge = challenge_response.json()
    message = presentation_signing_bytes(
        challenge_id=challenge["challenge_id"],
        nonce=challenge["nonce"],
        audience=challenge["audience"],
        credential_jti=metadata["jti"],
        expires_at=challenge["expires_at"],
    )

    copied = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": challenge["challenge_id"],
            "attestation": attestation,
            "holder_signature": _sign(copied_key, message),
        },
    )
    assert copied.status_code == 400
    assert "holder signature" in copied.json()["detail"].lower()

    valid = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": challenge["challenge_id"],
            "attestation": attestation,
            "holder_signature": _sign(holder_key, message),
        },
    )
    assert valid.status_code == 200, valid.text
    assert valid.json()["valid"] is True
    assert valid.json()["tier"] == "bronze"

    replay = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": challenge["challenge_id"],
            "attestation": attestation,
            "holder_signature": _sign(holder_key, message),
        },
    )
    assert replay.status_code == 400
    assert "expired or already used" in replay.json()["detail"].lower()


def test_presentation_fails_closed_when_credential_is_revoked(
    client: TestClient,
) -> None:
    holder_key, public_key_pem = _keypair()
    attestation = _complete_presence_session(client, holder_key, public_key_pem)
    metadata = attestation["metadata"]
    cast(FastAPI, client.app).state.credential_revocation_checker = lambda jti: (
        jti == metadata["jti"]
    )
    challenge = client.post(
        "/api/mettle/presentation-challenges",
        json={
            "credential_jti": metadata["jti"],
            "audience": "service.example",
        },
    ).json()
    message = presentation_signing_bytes(
        challenge_id=challenge["challenge_id"],
        nonce=challenge["nonce"],
        audience=challenge["audience"],
        credential_jti=metadata["jti"],
        expires_at=challenge["expires_at"],
    )
    response = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": challenge["challenge_id"],
            "attestation": attestation,
            "holder_signature": _sign(holder_key, message),
        },
    )
    assert response.status_code == 400
    assert "revoked" in response.json()["detail"].lower()


def test_submission_signatures_bind_answers_and_transcript() -> None:
    private_key, _ = _keypair()
    message = submission_signing_bytes(
        session_id="session-1",
        action="suite:adversarial",
        nonce="nonce-1",
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=answer_hash({"q1": 42}),
    )
    signature = _sign(private_key, message)
    assert signature
    assert message != submission_signing_bytes(
        session_id="session-1",
        action="suite:adversarial",
        nonce="nonce-1",
        previous_transcript_hash="sha256:" + "a" * 64,
        payload_hash=answer_hash({"q1": 43}),
    )
    assert json.loads(message)["protocol"] == "mettle-presence-v1"


def test_stale_nonce_and_answer_substitution_are_rejected(client: TestClient) -> None:
    private_key, public_key_pem = _keypair()
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["adversarial", "native"],
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    ).json()
    session_id = created["session_id"]
    initial = created["presence"]
    answers = _answers_with_continuity(created["challenges"], "adversarial", {"q1": 42})
    first_message = submission_signing_bytes(
        session_id=session_id,
        action="suite:adversarial",
        nonce=initial["nonce"],
        previous_transcript_hash=initial["transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    first = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={
            "suite": "adversarial",
            "answers": answers,
            "presence_proof": {
                "nonce": initial["nonce"],
                "previous_transcript_hash": initial["transcript_hash"],
                "signature": _sign(private_key, first_message),
            },
        },
    )
    assert first.status_code == 200
    current = first.json()["presence"]
    current_answers = _answers_with_continuity(
        first.json()["next_challenge"], "native", {"q1": 42}
    )

    stale_message = submission_signing_bytes(
        session_id=session_id,
        action="suite:native",
        nonce=initial["nonce"],
        previous_transcript_hash=initial["transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    stale = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={
            "suite": "native",
            "answers": answers,
            "presence_proof": {
                "nonce": initial["nonce"],
                "previous_transcript_hash": initial["transcript_hash"],
                "signature": _sign(private_key, stale_message),
            },
        },
    )
    assert stale.status_code == 400
    assert "already been used" in stale.json()["detail"].lower()

    signed_for_different_answers = submission_signing_bytes(
        session_id=session_id,
        action="suite:native",
        nonce=current["nonce"],
        previous_transcript_hash=current["transcript_hash"],
        payload_hash=answer_hash(current_answers),
    )
    substituted_answers = {**current_answers, "q1": 43}
    substituted = client.post(
        f"/api/mettle/sessions/{session_id}/verify",
        json={
            "suite": "native",
            "answers": substituted_answers,
            "presence_proof": {
                "nonce": current["nonce"],
                "previous_transcript_hash": current["transcript_hash"],
                "signature": _sign(private_key, signed_for_different_answers),
            },
        },
    )
    assert substituted.status_code == 400
    assert "signature" in substituted.json()["detail"].lower()


def test_unissued_suite_cannot_be_skipped_to(client: TestClient) -> None:
    private_key, public_key_pem = _keypair()
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["adversarial", "native"],
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    ).json()
    state = created["presence"]
    answers = _answers_with_continuity(created["challenges"], "adversarial", {"q1": 42})
    message = submission_signing_bytes(
        session_id=created["session_id"],
        action="suite:native",
        nonce=state["nonce"],
        previous_transcript_hash=state["transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    response = client.post(
        f"/api/mettle/sessions/{created['session_id']}/verify",
        json={
            "suite": "native",
            "answers": answers,
            "presence_proof": {
                "nonce": state["nonce"],
                "previous_transcript_hash": state["transcript_hash"],
                "signature": _sign(private_key, message),
            },
        },
    )

    assert response.status_code == 400
    assert "not currently issued" in response.json()["detail"].lower()


def test_audience_mismatch_and_attestation_tampering_are_rejected(
    client: TestClient,
) -> None:
    holder_key, public_key_pem = _keypair()
    attestation = _complete_presence_session(client, holder_key, public_key_pem)
    metadata = attestation["metadata"]

    tampered = copy.deepcopy(attestation)
    tampered["metadata"]["tier"] = "gold"
    challenge = client.post(
        "/api/mettle/presentation-challenges",
        json={
            "credential_jti": metadata["jti"],
            "audience": "service.example",
        },
    ).json()
    tampered_response = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": challenge["challenge_id"],
            "attestation": tampered,
            "holder_signature": _sign(holder_key, b"untrusted"),
        },
    )
    assert tampered_response.status_code == 400
    assert "credential signature" in tampered_response.json()["detail"].lower()

    wrong_audience = client.post(
        "/api/mettle/presentation-challenges",
        json={
            "credential_jti": metadata["jti"],
            "audience": "other.example",
        },
    ).json()
    message = presentation_signing_bytes(
        challenge_id=wrong_audience["challenge_id"],
        nonce=wrong_audience["nonce"],
        audience=wrong_audience["audience"],
        credential_jti=metadata["jti"],
        expires_at=wrong_audience["expires_at"],
    )
    response = client.post(
        "/api/mettle/presentations/verify",
        json={
            "challenge_id": wrong_audience["challenge_id"],
            "attestation": attestation,
            "holder_signature": _sign(holder_key, message),
        },
    )
    assert response.status_code == 400
    assert "audience" in response.json()["detail"].lower()


def test_presence_registration_rejects_non_ed25519_key(client: TestClient) -> None:
    response = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["adversarial"],
            "presence": {
                "public_key_pem": "not-a-public-key",
                "audience": "service.example",
            },
        },
    )
    assert response.status_code == 400
    assert "public key" in response.json()["detail"].lower()


def test_presentation_challenge_creation_is_rate_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_manager, "MAX_PRESENTATION_CHALLENGES_PER_MINUTE", 1)
    payload = {"credential_jti": "c" * 32, "audience": "service.example"}
    first = client.post("/api/mettle/presentation-challenges", json=payload)
    limited = client.post("/api/mettle/presentation-challenges", json=payload)

    assert first.status_code == 201
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_multi_round_answers_participate_in_the_same_transcript(
    client: TestClient,
) -> None:
    private_key, public_key_pem = _keypair()
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["novel-reasoning"],
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    ).json()
    state = created["presence"]
    answers = _answers_with_continuity(
        created["challenges"],
        "novel-reasoning",
        {"challenges": {"seq": {"value": 1}}},
    )
    message = submission_signing_bytes(
        session_id=created["session_id"],
        action="round:1",
        nonce=state["nonce"],
        previous_transcript_hash=state["transcript_hash"],
        payload_hash=answer_hash(answers),
    )
    response = client.post(
        f"/api/mettle/sessions/{created['session_id']}/rounds/1/answer",
        json={
            "answers": answers,
            "presence_proof": {
                "nonce": state["nonce"],
                "previous_transcript_hash": state["transcript_hash"],
                "signature": _sign(private_key, message),
            },
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["presence"]["completed"] is True
    assert response.json()["presence"]["sequence"] == 1
    assert response.json()["presence"]["action"] is None


def test_future_continuity_challenge_is_transcript_bound_and_unharvestable(
    client: TestClient,
) -> None:
    private_key, public_key_pem = _keypair()
    created = client.post(
        "/api/mettle/sessions",
        json={
            "suites": ["adversarial", "native"],
            "presence": {
                "public_key_pem": public_key_pem,
                "audience": "service.example",
            },
        },
    ).json()
    first_challenge = created["challenges"]["adversarial"][CONTINUITY_CHALLENGE_KEY]
    assert first_challenge["protocol"] == CONTINUITY_PROTOCOL
    assert "native" not in created["challenges"]

    first_answers = _answers_with_continuity(
        created["challenges"], "adversarial", {"q1": 42}
    )
    state = created["presence"]
    message = submission_signing_bytes(
        session_id=created["session_id"],
        action="suite:adversarial",
        nonce=state["nonce"],
        previous_transcript_hash=state["transcript_hash"],
        payload_hash=answer_hash(first_answers),
    )
    first = client.post(
        f"/api/mettle/sessions/{created['session_id']}/verify",
        json={
            "suite": "adversarial",
            "answers": first_answers,
            "presence_proof": {
                "nonce": state["nonce"],
                "previous_transcript_hash": state["transcript_hash"],
                "signature": _sign(private_key, message),
            },
        },
    )
    assert first.status_code == 200, first.text
    body = first.json()
    second_challenge = body["next_challenge"]["native"][CONTINUITY_CHALLENGE_KEY]
    assert second_challenge["challenge_id"] != first_challenge["challenge_id"]

    harvested_answers = {
        "q1": 42,
        CONTINUITY_ANSWER_KEY: {
            "challenge_id": first_challenge["challenge_id"],
            "computed": solve_continuity_challenge(first_challenge),
        },
    }
    current = body["presence"]
    harvested_message = submission_signing_bytes(
        session_id=created["session_id"],
        action="suite:native",
        nonce=current["nonce"],
        previous_transcript_hash=current["transcript_hash"],
        payload_hash=answer_hash(harvested_answers),
    )
    rejected = client.post(
        f"/api/mettle/sessions/{created['session_id']}/verify",
        json={
            "suite": "native",
            "answers": harvested_answers,
            "presence_proof": {
                "nonce": current["nonce"],
                "previous_transcript_hash": current["transcript_hash"],
                "signature": _sign(private_key, harvested_message),
            },
        },
    )
    assert rejected.status_code == 400
    assert "continuity challenge" in rejected.json()["detail"].lower()
