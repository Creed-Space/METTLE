"""Shared live-client support for METTLE Presence resilience trials.

This module intentionally uses only public API responses plus the holder's
private key. It never reads server answers or server-side session state.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from mettle.presence import (
    answer_hash,
    presentation_signing_bytes,
    submission_signing_bytes,
)
from mettle.solver import solve_suite
from mettle.vcp import verify_mettle_attestation

BRONZE_SUITES = [
    "adversarial",
    "native",
    "self-reference",
    "social",
    "inverse-turing",
]


class TrialFailure(RuntimeError):
    """A live trial failed to satisfy its expected API contract."""


def validate_trial_url(base_url: str) -> str:
    """Require TLS except for an explicitly local test server."""
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme == "https" and parsed.netloc:
        return normalized
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return normalized
    raise ValueError("Trial URL must use HTTPS unless it targets localhost")


def _public_key_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )


def ephemeral_vcp_signing_key_pem() -> str:
    """Create a process-local issuer key for an isolated resilience API."""
    return (
        Ed25519PrivateKey.generate()
        .private_bytes(
            Encoding.PEM,
            PrivateFormat.PKCS8,
            NoEncryption(),
        )
        .decode("ascii")
    )


def _signature(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


@dataclass
class SubmissionObservation:
    """Client-side measurements for one accepted suite submission."""

    action: str
    configured_delay_ms: int
    solve_time_ms: float
    request_time_ms: float


@dataclass
class PresenceSessionDriver:
    """Drive one key-bound Presence session through the public REST API."""

    base_url: str
    api_key: str
    suites: list[str] = field(default_factory=lambda: list(BRONZE_SUITES))
    audience: str = "presence-trial.local"
    timeout_seconds: float = 30.0
    private_key: Ed25519PrivateKey = field(
        default_factory=Ed25519PrivateKey.generate, init=False
    )
    session_id: str | None = field(default=None, init=False)
    presence: dict[str, Any] | None = field(default=None, init=False)
    current_challenges: dict[str, Any] = field(default_factory=dict, init=False)
    observations: list[SubmissionObservation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.base_url = validate_trial_url(self.base_url)
        if not self.api_key:
            raise ValueError("A non-empty METTLE API key is required")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PresenceSessionDriver:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def public_key_pem(self) -> str:
        return _public_key_pem(self.private_key)

    def request(
        self,
        method: str,
        path: str,
        *,
        expected_status: int | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        response = self._client.request(method, path, **kwargs)
        if expected_status is not None and response.status_code != expected_status:
            detail = response.text[:1000]
            raise TrialFailure(
                f"{method} {path} returned {response.status_code}, expected "
                f"{expected_status}: {detail}"
            )
        return response

    def start(self) -> dict[str, Any]:
        response = self.request(
            "POST",
            "/api/mettle/sessions",
            expected_status=201,
            json={
                "suites": self.suites,
                "presence": {
                    "public_key_pem": self.public_key_pem,
                    "audience": self.audience,
                },
            },
        )
        body = response.json()
        self.session_id = body["session_id"]
        self.presence = body["presence"]
        self.current_challenges = body["challenges"]
        return body

    def submit_current(self, *, delay_ms: int = 0) -> dict[str, Any]:
        if self.session_id is None or self.presence is None:
            raise TrialFailure("Presence session has not been started")
        if delay_ms < 0:
            raise ValueError("Configured delay must be non-negative")
        if len(self.current_challenges) != 1:
            raise TrialFailure("Presence API must expose exactly one current suite")

        suite, challenge = next(iter(self.current_challenges.items()))
        action = f"suite:{suite}"
        solve_started = time.perf_counter()
        answers = solve_suite(suite, challenge)
        solve_time_ms = (time.perf_counter() - solve_started) * 1000
        if delay_ms:
            time.sleep(delay_ms / 1000)

        message = submission_signing_bytes(
            session_id=self.session_id,
            action=action,
            nonce=self.presence["nonce"],
            previous_transcript_hash=self.presence["transcript_hash"],
            payload_hash=answer_hash(answers),
        )
        payload = {
            "suite": suite,
            "answers": answers,
            "presence_proof": {
                "nonce": self.presence["nonce"],
                "previous_transcript_hash": self.presence["transcript_hash"],
                "signature": _signature(self.private_key, message),
            },
        }
        request_started = time.perf_counter()
        response = self.request(
            "POST",
            f"/api/mettle/sessions/{self.session_id}/verify",
            expected_status=200,
            json=payload,
        )
        request_time_ms = (time.perf_counter() - request_started) * 1000
        body = response.json()
        if body.get("passed") is not True:
            raise TrialFailure(f"Reference solver did not pass {suite}")
        self.presence = body["presence"]
        self.current_challenges = body.get("next_challenge") or {}
        self.observations.append(
            SubmissionObservation(
                action=action,
                configured_delay_ms=delay_ms,
                solve_time_ms=round(solve_time_ms, 3),
                request_time_ms=round(request_time_ms, 3),
            )
        )
        return body

    def complete(self, *, delay_ms: int = 0) -> dict[str, Any]:
        if self.session_id is None:
            self.start()
        while self.current_challenges:
            self.submit_current(delay_ms=delay_ms)
        return self.result()

    def status(self, *, expected_status: int = 200) -> httpx.Response:
        if self.session_id is None:
            raise TrialFailure("Presence session has not been started")
        return self.request(
            "GET",
            f"/api/mettle/sessions/{self.session_id}",
            expected_status=expected_status,
        )

    def result(self) -> dict[str, Any]:
        if self.session_id is None:
            raise TrialFailure("Presence session has not been started")
        response = self.request(
            "GET",
            f"/api/mettle/sessions/{self.session_id}/result?include_vcp=true",
            expected_status=200,
        )
        body = response.json()
        attestation = body.get("vcp_attestation")
        key_info = self.request(
            "GET",
            "/api/mettle/.well-known/vcp-keys",
            expected_status=200,
        ).json()
        public_key_pem = key_info.get("public_key_pem")
        if (
            not isinstance(attestation, dict)
            or not isinstance(public_key_pem, str)
            or not verify_mettle_attestation(attestation, public_key_pem)
        ):
            raise TrialFailure("METTLE issuer signature verification failed")
        return body

    def create_presentation_challenge(self, credential_jti: str) -> dict[str, Any]:
        response = self.request(
            "POST",
            "/api/mettle/presentation-challenges",
            expected_status=201,
            json={"credential_jti": credential_jti, "audience": self.audience},
        )
        return response.json()

    def verify_presentation(
        self,
        challenge: dict[str, Any],
        credential_jti: str,
        attestation: dict[str, Any],
        *,
        expected_status: int = 200,
    ) -> httpx.Response:
        message = presentation_signing_bytes(
            challenge_id=challenge["challenge_id"],
            nonce=challenge["nonce"],
            audience=challenge["audience"],
            credential_jti=credential_jti,
            expires_at=challenge["expires_at"],
        )
        return self.request(
            "POST",
            "/api/mettle/presentations/verify",
            expected_status=expected_status,
            json={
                "challenge_id": challenge["challenge_id"],
                "attestation": attestation,
                "holder_signature": _signature(self.private_key, message),
            },
        )


def presence_timing_receipts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract issuer-signed server timing receipts from a completed result."""
    try:
        attestation = result["vcp_attestation"]
        if attestation.get("credential_issued") is not True:
            raise KeyError("credential_issued")
        return attestation["metadata"]["proof_of_possession"]["server_timing"][
            "submissions"
        ]
    except (AttributeError, KeyError, TypeError) as exc:
        raise TrialFailure(
            "Completed session lacks signed Presence timing receipts"
        ) from exc
