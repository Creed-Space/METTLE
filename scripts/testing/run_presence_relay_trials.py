#!/usr/bin/env python3
"""Run three-party Presence relay and credential-copy trials.

The default smoke run exercises four autonomous cohorts against a live METTLE
deployment: an in-process control, a real holder/solver subprocess relay, that
relay with a synthetic holder-service delay, and a paced relay. Manual mode is
available for optional calibration, but it is never required by the automated
security decision.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mettle.presence import (
    answer_hash,
    presentation_signing_bytes,
    submission_signing_bytes,
)
from mettle.vcp import verify_mettle_attestation
from scripts.testing.presence_relay_workers import (
    HolderWorkerClient,
    SolverWorkerClient,
    WorkerProtocolError,
)
from scripts.testing.presence_trial_support import (
    BRONZE_SUITES,
    PresenceSessionDriver,
    TrialFailure,
    presence_timing_receipts,
    validate_trial_url,
)


COHORT_KINDS = {
    "direct",
    "process-relay",
    "holder-service",
    "paced-relay",
    "synthetic-human",
    "manual-human",
}
MAX_DIRECT_FALSE_POSITIVE_RATE = 0.01
MIN_HUMAN_RELAY_DETECTION_RATE = 0.90
MINIMUM_DIRECT_RECEIPTS = 300
MINIMUM_MEASURED_HUMAN_RECEIPTS = 29
REQUIRED_HOLDER_POLICY_ATTACKS = frozenset(
    {
        "unauthorized_session_signing",
        "unregistered_credential_presentation",
        "state_receipt_session_substitution",
        "state_receipt_field_tampering",
        "active_session_budget",
        "action_substitution",
        "transcript_rollback_signing",
        "pending_submission_fork",
        "state_rollback_commit",
        "committed_submission_replay",
        "tampered_credential_registration",
        "presentation_audience_substitution",
        "presentation_challenge_fork",
        "presentation_budget",
    }
)
REQUIRED_PROTOCOL_REJECTIONS = frozenset(
    {
        "stolen_session_without_holder_key",
        "harvested_submission_replay",
        "tier_tampering",
        "copied_credential_without_holder_key",
        "presentation_replay",
    }
)
AUTOMATED_SECURITY_CRITERIA = {
    "required_holder_policy_rejections": sorted(REQUIRED_HOLDER_POLICY_ATTACKS),
    "required_protocol_rejections": sorted(REQUIRED_PROTOCOL_REJECTIONS),
    "require_valid_holder_service_presentation": True,
    "require_isolated_holder_key": True,
    "require_workers_without_mettle_credentials": True,
    "human_testing_required": False,
    "timing_threshold_enforcement_authorized": False,
}
OPTIONAL_TIMING_CALIBRATION_CRITERIA = {
    "status": "optional_non_blocking_calibration",
    "max_direct_false_positive_rate": MAX_DIRECT_FALSE_POSITIVE_RATE,
    "min_human_relay_detection_rate": MIN_HUMAN_RELAY_DETECTION_RATE,
    "require_no_cohort_overlap": True,
    "minimum_direct_receipts": MINIMUM_DIRECT_RECEIPTS,
    "minimum_measured_human_receipts": MINIMUM_MEASURED_HUMAN_RECEIPTS,
    "authorizes_product_gating": False,
    "confidence_note": (
        "With zero observed errors, 300 direct receipts bound a 95% one-sided "
        "false-positive rate near 1%, and 29 measured relay receipts bound a "
        "95% one-sided detection rate above 90%."
    ),
}


@dataclass(frozen=True)
class Cohort:
    name: str
    kind: str
    delay_ms: int
    sessions: int

    @property
    def is_measured_human(self) -> bool:
        return self.kind == "manual-human"

    @property
    def is_synthetic(self) -> bool:
        return (
            self.kind
            in {
                "holder-service",
                "paced-relay",
                "synthetic-human",
            }
            and self.delay_ms > 0
        )


@dataclass
class PreparedSubmission:
    suite: str
    action: str
    answers: dict[str, Any]
    solve_time_ms: float
    solver_roundtrip_ms: float


@dataclass
class RelayObservation:
    action: str
    configured_delay_ms: int
    solve_time_ms: float
    solver_roundtrip_ms: float
    holder_roundtrip_ms: float
    request_time_ms: float
    solver_pid: int | None
    holder_pid: int


class Solver(Protocol):
    @property
    def pid(self) -> int | None: ...

    def solve(
        self, suite: str, challenge: dict[str, Any]
    ) -> tuple[dict[str, Any], float, float]: ...


class ManualSolver:
    """Interactive client-visible challenge handoff for a measured human trial."""

    pid: int | None = None

    def solve(
        self, suite: str, challenge: dict[str, Any]
    ) -> tuple[dict[str, Any], float, float]:
        print(
            json.dumps({"suite": suite, "challenge": challenge}, indent=2),
            file=sys.stderr,
        )
        print("Enter the complete answers object as one JSON line:", file=sys.stderr)
        started = time.perf_counter()
        line = sys.stdin.readline()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not line:
            raise TrialFailure("Manual human trial ended before an answer was entered")
        try:
            answers = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrialFailure("Manual human answer was not valid JSON") from exc
        if not isinstance(answers, dict):
            raise TrialFailure("Manual human answer must be a JSON object")
        return answers, round(elapsed_ms, 3), round(elapsed_ms, 3)


@dataclass
class RelayedPresenceSessionDriver:
    """Drive a session while the holder and solver remain separate principals."""

    base_url: str
    api_key: str
    holder: HolderWorkerClient
    solver: Solver
    relay_delay_ms: int = 0
    suites: list[str] = field(default_factory=lambda: list(BRONZE_SUITES))
    audience: str = "relay-trial.mettle.local"
    timeout_seconds: float = 30.0
    session_id: str | None = field(default=None, init=False)
    presence: dict[str, Any] | None = field(default=None, init=False)
    current_challenges: dict[str, Any] = field(default_factory=dict, init=False)
    observations: list[RelayObservation] = field(default_factory=list, init=False)
    last_submission_payload: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.base_url = validate_trial_url(self.base_url)
        if not self.api_key:
            raise ValueError("A non-empty METTLE API key is required")
        if self.relay_delay_ms < 0:
            raise ValueError("Relay delay must be non-negative")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )
        key_info = self.request(
            "GET", "/api/mettle/.well-known/vcp-keys", expected_status=200
        ).json()
        issuer_public_key_pem = key_info.get("public_key_pem")
        if not isinstance(issuer_public_key_pem, str):
            raise TrialFailure("METTLE issuer key is unavailable")
        self._issuer_public_key_pem = issuer_public_key_pem
        self.holder.configure(
            issuer=self.base_url,
            issuer_public_key_pem=issuer_public_key_pem,
            allowed_audiences=[self.audience],
            max_active_sessions=1,
            max_actions_per_session=max(16, len(self.suites) + 5),
            max_presentations_per_credential=1,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RelayedPresenceSessionDriver:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

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
            raise TrialFailure(
                f"{method} {path} returned {response.status_code}, expected "
                f"{expected_status}: {response.text[:1000]}"
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
                    "public_key_pem": self.holder.public_key_pem,
                    "audience": self.audience,
                },
            },
        )
        body = response.json()
        self.session_id = body["session_id"]
        self.presence = body["presence"]
        self.current_challenges = body["challenges"]
        self.holder.authorize_session(
            issuer=self.base_url,
            session_id=self.session_id,
            presence=self.presence,
        )
        return body

    def prepare_current(self) -> PreparedSubmission:
        if self.session_id is None or self.presence is None:
            raise TrialFailure("Relayed Presence session has not been started")
        if len(self.current_challenges) != 1:
            raise TrialFailure("Presence API must expose exactly one current suite")
        suite, challenge = next(iter(self.current_challenges.items()))
        answers, solve_time_ms, solver_roundtrip_ms = self.solver.solve(
            suite, challenge
        )
        return PreparedSubmission(
            suite=suite,
            action=f"suite:{suite}",
            answers=answers,
            solve_time_ms=round(solve_time_ms, 3),
            solver_roundtrip_ms=round(solver_roundtrip_ms, 3),
        )

    def payload_for(
        self, prepared: PreparedSubmission, signature: str
    ) -> dict[str, Any]:
        if self.presence is None:
            raise TrialFailure("Relayed Presence session has no current state")
        return {
            "suite": prepared.suite,
            "answers": prepared.answers,
            "presence_proof": {
                "nonce": self.presence["nonce"],
                "previous_transcript_hash": self.presence["transcript_hash"],
                "signature": signature,
            },
        }

    def submit_prepared(
        self,
        prepared: PreparedSubmission,
        signature: str,
        *,
        holder_roundtrip_ms: float,
    ) -> dict[str, Any]:
        if self.session_id is None:
            raise TrialFailure("Relayed Presence session has not been started")
        payload = self.payload_for(prepared, signature)
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
            raise TrialFailure(f"Reference relay solver did not pass {prepared.suite}")
        self.holder.commit_submission(
            session_id=self.session_id,
            presence=body["presence"],
        )
        self.last_submission_payload = copy.deepcopy(payload)
        self.presence = body["presence"]
        self.current_challenges = body.get("next_challenge") or {}
        self.observations.append(
            RelayObservation(
                action=prepared.action,
                configured_delay_ms=self.relay_delay_ms,
                solve_time_ms=prepared.solve_time_ms,
                solver_roundtrip_ms=prepared.solver_roundtrip_ms,
                holder_roundtrip_ms=round(holder_roundtrip_ms, 3),
                request_time_ms=round(request_time_ms, 3),
                solver_pid=self.solver.pid,
                holder_pid=self.holder.pid,
            )
        )
        return body

    def submit_current(self) -> dict[str, Any]:
        if self.session_id is None or self.presence is None:
            raise TrialFailure("Relayed Presence session has not been started")
        prepared = self.prepare_current()
        if self.relay_delay_ms:
            time.sleep(self.relay_delay_ms / 1000)
        signature, holder_roundtrip_ms = self.holder.sign_submission(
            session_id=self.session_id,
            action=prepared.action,
            nonce=self.presence["nonce"],
            previous_transcript_hash=self.presence["transcript_hash"],
            payload_hash=answer_hash(prepared.answers),
        )
        return self.submit_prepared(
            prepared, signature, holder_roundtrip_ms=holder_roundtrip_ms
        )

    def complete(self) -> dict[str, Any]:
        if self.session_id is None:
            self.start()
        while self.current_challenges:
            self.submit_current()
        return self.result()

    def result(self) -> dict[str, Any]:
        if self.session_id is None:
            raise TrialFailure("Relayed Presence session has not been started")
        body = self.request(
            "GET",
            f"/api/mettle/sessions/{self.session_id}/result?include_vcp=true",
            expected_status=200,
        ).json()
        key_info = self.request(
            "GET", "/api/mettle/.well-known/vcp-keys", expected_status=200
        ).json()
        attestation = body.get("vcp_attestation")
        public_key_pem = key_info.get("public_key_pem")
        if (
            not isinstance(attestation, dict)
            or not isinstance(public_key_pem, str)
            or not verify_mettle_attestation(attestation, public_key_pem)
        ):
            raise TrialFailure("METTLE issuer signature verification failed")
        registered_jti = self.holder.register_credential(
            issuer=self.base_url,
            attestation=attestation,
        )
        if registered_jti != attestation["metadata"]["jti"]:
            raise TrialFailure("Holder registered a different credential JTI")
        return body


def parse_cohort(value: str) -> Cohort:
    """Parse NAME:KIND:DELAY_MS:SESSIONS."""
    try:
        name, kind, delay_text, sessions_text = value.split(":", 3)
        delay_ms = int(delay_text)
        sessions = int(sessions_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "cohort must use NAME:KIND:DELAY_MS:SESSIONS"
        ) from exc
    if not name or kind not in COHORT_KINDS or delay_ms < 0 or sessions < 1:
        raise argparse.ArgumentTypeError(
            "cohort requires a name, supported kind, non-negative delay, and sessions"
        )
    if kind == "direct" and delay_ms != 0:
        raise argparse.ArgumentTypeError("direct cohorts cannot inject delay")
    if kind == "manual-human" and delay_ms != 0:
        raise argparse.ArgumentTypeError("manual-human cohorts record observed delay")
    return Cohort(name=name, kind=kind, delay_ms=delay_ms, sessions=sessions)


def _percentile(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Timing sample cannot be empty")
    rank = max(1, int(len(ordered) * proportion + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Timing sample cannot be empty")
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def evaluate_separation(
    direct_values: list[int], relay_values: list[int], *, measured_human: bool
) -> dict[str, Any]:
    """Evaluate observed timing against the criteria fixed before this run."""
    if not direct_values or not relay_values:
        raise ValueError("Direct and relay timing samples are required")
    qualified: list[dict[str, int | float]] = []
    for threshold in sorted(set(direct_values + relay_values)):
        false_positive_rate = sum(value > threshold for value in direct_values) / len(
            direct_values
        )
        detection_rate = sum(value > threshold for value in relay_values) / len(
            relay_values
        )
        if (
            false_positive_rate <= MAX_DIRECT_FALSE_POSITIVE_RATE
            and detection_rate >= MIN_HUMAN_RELAY_DETECTION_RATE
        ):
            qualified.append(
                {
                    "threshold_ms": threshold,
                    "direct_false_positive_rate": round(false_positive_rate, 6),
                    "relay_detection_rate": round(detection_rate, 6),
                }
            )
    no_overlap = max(direct_values) < min(relay_values)
    sample_sufficient = (
        len(direct_values) >= MINIMUM_DIRECT_RECEIPTS
        and len(relay_values) >= MINIMUM_MEASURED_HUMAN_RECEIPTS
    )
    return {
        "direct_max_to_relay_min_gap_ms": min(relay_values) - max(direct_values),
        "observed_overlap": not no_overlap,
        "best_observed_threshold": qualified[0] if qualified and no_overlap else None,
        "observed_rate_criteria_met": bool(qualified and no_overlap),
        "measured_human": measured_human,
        "sample_sufficient": sample_sufficient,
        "optional_human_calibration_eligible": bool(
            qualified and no_overlap and measured_human and sample_sufficient
        ),
    }


def _complete_with_rate_retry(driver: Any) -> dict[str, Any]:
    for attempt in range(2):
        try:
            return driver.complete()
        except TrialFailure as exc:
            if "returned 429" not in str(exc) or attempt:
                raise
            time.sleep(61)
    raise AssertionError("rate retry loop exhausted")


def _run_direct_session(
    *, base_url: str, api_key: str, cohort: Cohort, session_index: int, timeout: float
) -> list[dict[str, Any]]:
    with PresenceSessionDriver(
        base_url=base_url,
        api_key=api_key,
        suites=list(BRONZE_SUITES),
        audience="relay-control.mettle.local",
        timeout_seconds=timeout,
    ) as driver:
        result = _complete_with_rate_retry(driver)
        receipts = presence_timing_receipts(result)
        if len(receipts) != len(driver.observations):
            raise TrialFailure("Direct client and signed timing receipts differ")
        return [
            {
                "cohort": cohort.name,
                "kind": cohort.kind,
                "session_index": session_index,
                "action": observation.action,
                "configured_delay_ms": observation.configured_delay_ms,
                "client_solve_time_ms": observation.solve_time_ms,
                "solver_roundtrip_ms": observation.solve_time_ms,
                "holder_roundtrip_ms": 0.0,
                "client_request_time_ms": observation.request_time_ms,
                "server_response_time_ms": receipt["response_time_ms"],
                "solver_pid": os.getpid(),
                "holder_pid": os.getpid(),
                "orchestrator_pid": os.getpid(),
                "process_isolated": False,
            }
            for observation, receipt in zip(driver.observations, receipts, strict=True)
        ]


def _run_relay_session(
    *, base_url: str, api_key: str, cohort: Cohort, session_index: int, timeout: float
) -> list[dict[str, Any]]:
    with HolderWorkerClient(timeout_seconds=timeout) as holder:
        solver_context: SolverWorkerClient | None = None
        solver: Solver
        if cohort.kind == "manual-human":
            solver = ManualSolver()
        else:
            solver_context = SolverWorkerClient(timeout_seconds=timeout)
            solver = solver_context
        try:
            with RelayedPresenceSessionDriver(
                base_url=base_url,
                api_key=api_key,
                holder=holder,
                solver=solver,
                relay_delay_ms=cohort.delay_ms,
                timeout_seconds=timeout,
            ) as driver:
                result = _complete_with_rate_retry(driver)
                receipts = presence_timing_receipts(result)
                if len(receipts) != len(driver.observations):
                    raise TrialFailure("Relay client and signed timing receipts differ")
                return [
                    {
                        "cohort": cohort.name,
                        "kind": cohort.kind,
                        "session_index": session_index,
                        "action": observation.action,
                        "configured_delay_ms": observation.configured_delay_ms,
                        "client_solve_time_ms": observation.solve_time_ms,
                        "solver_roundtrip_ms": observation.solver_roundtrip_ms,
                        "holder_roundtrip_ms": observation.holder_roundtrip_ms,
                        "client_request_time_ms": observation.request_time_ms,
                        "server_response_time_ms": receipt["response_time_ms"],
                        "solver_pid": observation.solver_pid,
                        "holder_pid": observation.holder_pid,
                        "orchestrator_pid": os.getpid(),
                        "process_isolated": True,
                    }
                    for observation, receipt in zip(
                        driver.observations, receipts, strict=True
                    )
                ]
        finally:
            if solver_context is not None:
                solver_context.close()


def _signature(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


def run_attack_trials(*, base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    """Exercise theft, harvesting, tier tamper, copying, and replay boundaries."""
    wrong_key = Ed25519PrivateKey.generate()
    with (
        HolderWorkerClient(timeout_seconds=timeout) as holder,
        SolverWorkerClient(timeout_seconds=timeout) as solver,
        RelayedPresenceSessionDriver(
            base_url=base_url,
            api_key=api_key,
            holder=holder,
            solver=solver,
            timeout_seconds=timeout,
        ) as driver,
    ):
        holder_policy_attacks: dict[str, dict[str, Any]] = {}

        def expect_policy_rejection(
            name: str,
            operation: Any,
            expected_fragments: tuple[str, ...],
        ) -> None:
            try:
                operation()
            except WorkerProtocolError as exc:
                error = str(exc)
                if not any(
                    fragment in error.lower() for fragment in expected_fragments
                ):
                    raise TrialFailure(
                        f"Holder rejected {name} for an unexpected reason: {error}"
                    ) from exc
                holder_policy_attacks[name] = {
                    "rejected": True,
                    "error": error,
                }
            else:
                raise TrialFailure(f"Holder accepted forbidden {name}")

        valid_expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        expect_policy_rejection(
            "unauthorized_session_signing",
            lambda: holder.sign_submission(
                session_id="unauthorized-session",
                action="suite:adversarial",
                nonce="u" * 32,
                previous_transcript_hash="sha256:" + "0" * 64,
                payload_hash="sha256:" + "1" * 64,
            ),
            ("not authorized",),
        )
        expect_policy_rejection(
            "unregistered_credential_presentation",
            lambda: holder.sign_presentation(
                challenge_id="unregistered-challenge",
                nonce="u" * 32,
                audience=driver.audience,
                credential_jti="0" * 32,
                expires_at=valid_expiry,
            ),
            ("not registered",),
        )

        driver.start()
        if driver.session_id is None or driver.presence is None:
            raise TrialFailure("Attack session did not initialize")
        expect_policy_rejection(
            "state_receipt_session_substitution",
            lambda: holder.authorize_session(
                issuer=driver.base_url,
                session_id="farmed-session",
                presence=copy.deepcopy(driver.presence),
            ),
            ("issuer receipt is invalid",),
        )
        tampered_state = copy.deepcopy(driver.presence)
        tampered_state["action"] = "round:999"
        expect_policy_rejection(
            "state_receipt_field_tampering",
            lambda: holder.authorize_session(
                issuer=driver.base_url,
                session_id=driver.session_id,
                presence=tampered_state,
            ),
            ("issuer receipt is invalid",),
        )
        farmed = driver.request(
            "POST",
            "/api/mettle/sessions",
            expected_status=201,
            json={
                "suites": [driver.suites[0]],
                "presence": {
                    "public_key_pem": holder.public_key_pem,
                    "audience": driver.audience,
                },
            },
        ).json()
        expect_policy_rejection(
            "active_session_budget",
            lambda: holder.authorize_session(
                issuer=driver.base_url,
                session_id=farmed["session_id"],
                presence=farmed["presence"],
            ),
            ("budget is exhausted",),
        )
        prepared = driver.prepare_current()
        expect_policy_rejection(
            "action_substitution",
            lambda: holder.sign_submission(
                session_id=driver.session_id,
                action="round:999",
                nonce=driver.presence["nonce"],
                previous_transcript_hash=driver.presence["transcript_hash"],
                payload_hash=answer_hash(prepared.answers),
            ),
            ("action does not match",),
        )
        expect_policy_rejection(
            "transcript_rollback_signing",
            lambda: holder.sign_submission(
                session_id=driver.session_id,
                action=prepared.action,
                nonce=driver.presence["nonce"],
                previous_transcript_hash="sha256:" + "0" * 64,
                payload_hash=answer_hash(prepared.answers),
            ),
            ("transcript does not match",),
        )
        message = submission_signing_bytes(
            session_id=driver.session_id,
            action=prepared.action,
            nonce=driver.presence["nonce"],
            previous_transcript_hash=driver.presence["transcript_hash"],
            payload_hash=answer_hash(prepared.answers),
        )
        stolen_payload = driver.payload_for(prepared, _signature(wrong_key, message))
        stolen = driver.request(
            "POST",
            f"/api/mettle/sessions/{driver.session_id}/verify",
            expected_status=400,
            json=stolen_payload,
        )
        if "signature" not in stolen.text.lower():
            raise TrialFailure(
                "Stolen-session rejection did not identify the key proof"
            )

        valid_signature, holder_ms = holder.sign_submission(
            session_id=driver.session_id,
            action=prepared.action,
            nonce=driver.presence["nonce"],
            previous_transcript_hash=driver.presence["transcript_hash"],
            payload_hash=answer_hash(prepared.answers),
        )
        signed_state = copy.deepcopy(driver.presence)
        expect_policy_rejection(
            "pending_submission_fork",
            lambda: holder.sign_submission(
                session_id=driver.session_id,
                action=prepared.action,
                nonce=signed_state["nonce"],
                previous_transcript_hash=signed_state["transcript_hash"],
                payload_hash="sha256:" + "f" * 64,
            ),
            ("different submission is already pending",),
        )
        expect_policy_rejection(
            "state_rollback_commit",
            lambda: holder.commit_submission(
                session_id=driver.session_id,
                presence=signed_state,
            ),
            ("did not advance exactly once",),
        )
        driver.submit_prepared(prepared, valid_signature, holder_roundtrip_ms=holder_ms)
        expect_policy_rejection(
            "committed_submission_replay",
            lambda: holder.sign_submission(
                session_id=driver.session_id,
                action=prepared.action,
                nonce=signed_state["nonce"],
                previous_transcript_hash=signed_state["transcript_hash"],
                payload_hash=answer_hash(prepared.answers),
            ),
            ("action does not match", "already complete"),
        )
        replay_payload = copy.deepcopy(driver.last_submission_payload)
        if replay_payload is None:
            raise TrialFailure(
                "Accepted submission payload was not retained for replay"
            )
        harvested_replay = driver.request(
            "POST",
            f"/api/mettle/sessions/{driver.session_id}/verify",
            expected_status=400,
            json=replay_payload,
        )
        result = _complete_with_rate_retry(driver)
        attestation = result["vcp_attestation"]
        if attestation.get("credential_issued") is not True:
            raise TrialFailure("Attack trial did not earn a bound credential")
        credential_jti = attestation["metadata"]["jti"]
        tampered_holder_attestation = copy.deepcopy(attestation)
        tampered_holder_attestation["metadata"]["tier"] = "platinum"
        expect_policy_rejection(
            "tampered_credential_registration",
            lambda: holder.register_credential(
                issuer=driver.base_url,
                attestation=tampered_holder_attestation,
            ),
            ("issuer signature or policy is invalid",),
        )
        challenge = driver.request(
            "POST",
            "/api/mettle/presentation-challenges",
            expected_status=201,
            json={
                "credential_jti": credential_jti,
                "audience": "relay-trial.mettle.local",
            },
        ).json()
        presentation_message = presentation_signing_bytes(
            challenge_id=challenge["challenge_id"],
            nonce=challenge["nonce"],
            audience=challenge["audience"],
            credential_jti=credential_jti,
            expires_at=challenge["expires_at"],
        )
        holder_signature, presentation_holder_ms = holder.sign_presentation(
            challenge_id=challenge["challenge_id"],
            nonce=challenge["nonce"],
            audience=challenge["audience"],
            credential_jti=credential_jti,
            expires_at=challenge["expires_at"],
        )
        expect_policy_rejection(
            "presentation_audience_substitution",
            lambda: holder.sign_presentation(
                challenge_id=f"{challenge['challenge_id']}-audience",
                nonce=challenge["nonce"],
                audience="wrong-audience.mettle.local",
                credential_jti=credential_jti,
                expires_at=challenge["expires_at"],
            ),
            ("audience is not allowed",),
        )
        expect_policy_rejection(
            "presentation_challenge_fork",
            lambda: holder.sign_presentation(
                challenge_id=challenge["challenge_id"],
                nonce="f" * 32,
                audience=challenge["audience"],
                credential_jti=credential_jti,
                expires_at=challenge["expires_at"],
            ),
            ("reused inconsistently",),
        )
        expect_policy_rejection(
            "presentation_budget",
            lambda: holder.sign_presentation(
                challenge_id=f"{challenge['challenge_id']}-budget",
                nonce="b" * 32,
                audience=challenge["audience"],
                credential_jti=credential_jti,
                expires_at=challenge["expires_at"],
            ),
            ("budget is exhausted",),
        )

        tampered_attestation = copy.deepcopy(attestation)
        tampered_attestation["metadata"]["tier"] = "platinum"
        tier_skip = driver.request(
            "POST",
            "/api/mettle/presentations/verify",
            expected_status=400,
            json={
                "challenge_id": challenge["challenge_id"],
                "attestation": tampered_attestation,
                "holder_signature": holder_signature,
            },
        )
        copied = driver.request(
            "POST",
            "/api/mettle/presentations/verify",
            expected_status=400,
            json={
                "challenge_id": challenge["challenge_id"],
                "attestation": attestation,
                "holder_signature": _signature(wrong_key, presentation_message),
            },
        )
        valid_payload = {
            "challenge_id": challenge["challenge_id"],
            "attestation": attestation,
            "holder_signature": holder_signature,
        }
        accepted = driver.request(
            "POST",
            "/api/mettle/presentations/verify",
            expected_status=200,
            json=valid_payload,
        )
        replay = driver.request(
            "POST",
            "/api/mettle/presentations/verify",
            expected_status=400,
            json=valid_payload,
        )
        return {
            "passed": True,
            "holder_policy_attacks": holder_policy_attacks,
            "process_boundary": {
                "orchestrator_pid": os.getpid(),
                "holder_pid": holder.pid,
                "solver_pid": solver.pid,
                "holder_private_key_crossed_ipc": False,
                "solver_received_holder_key": False,
                "workers_inherited_mettle_credentials": False,
            },
            "stolen_session_without_holder_key": {
                "http_status": stolen.status_code,
                "rejected": True,
            },
            "harvested_submission_replay": {
                "http_status": harvested_replay.status_code,
                "rejected": True,
            },
            "tier_tampering": {
                "http_status": tier_skip.status_code,
                "rejected": True,
            },
            "copied_credential_without_holder_key": {
                "http_status": copied.status_code,
                "rejected": True,
            },
            "valid_holder_service_presentation": {
                "http_status": accepted.status_code,
                "accepted": True,
                "holder_roundtrip_ms": presentation_holder_ms,
            },
            "presentation_replay": {
                "http_status": replay.status_code,
                "rejected": True,
            },
            "residual": (
                "A cooperating holder signing service and solver can complete the "
                "protocol. Key possession prevents copying and theft without the "
                "holder, but does not prove the signer and solver share a process."
            ),
        }


def build_report(
    *,
    base_url: str,
    cohorts: list[Cohort],
    samples: list[dict[str, Any]],
    attacks: dict[str, Any] | None,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    direct = next((cohort for cohort in cohorts if cohort.kind == "direct"), None)
    if direct is None:
        raise ValueError("At least one direct cohort is required")
    direct_values = [
        int(sample["server_response_time_ms"])
        for sample in samples
        if sample["cohort"] == direct.name
    ]
    separation: dict[str, Any] = {}
    for cohort in cohorts:
        values = [
            int(sample["server_response_time_ms"])
            for sample in samples
            if sample["cohort"] == cohort.name
        ]
        summaries[cohort.name] = {
            "kind": cohort.kind,
            "configured_delay_ms": cohort.delay_ms,
            "synthetic_delay": cohort.is_synthetic,
            "measured_human": cohort.is_measured_human,
            **summarize(values),
        }
        if cohort is not direct:
            separation[cohort.name] = evaluate_separation(
                direct_values, values, measured_human=cohort.is_measured_human
            )

    measured_human_present = any(cohort.is_measured_human for cohort in cohorts)
    holder_policy_attacks = (
        attacks.get("holder_policy_attacks") if isinstance(attacks, dict) else None
    )
    process_boundary = (
        attacks.get("process_boundary") if isinstance(attacks, dict) else None
    )
    holder_policy_attacks_passed = bool(
        isinstance(holder_policy_attacks, dict)
        and REQUIRED_HOLDER_POLICY_ATTACKS.issubset(holder_policy_attacks)
        and all(
            isinstance(holder_policy_attacks[name], dict)
            and holder_policy_attacks[name].get("rejected") is True
            for name in REQUIRED_HOLDER_POLICY_ATTACKS
        )
    )
    protocol_rejections_passed = bool(
        isinstance(attacks, dict)
        and all(
            isinstance(attacks.get(name), dict)
            and attacks[name].get("rejected") is True
            for name in REQUIRED_PROTOCOL_REJECTIONS
        )
    )
    valid_presentation = (
        attacks.get("valid_holder_service_presentation")
        if isinstance(attacks, dict)
        else None
    )
    process_boundary_passed = bool(
        isinstance(process_boundary, dict)
        and process_boundary.get("holder_private_key_crossed_ipc") is False
        and process_boundary.get("solver_received_holder_key") is False
        and process_boundary.get("workers_inherited_mettle_credentials") is False
    )
    attacks_passed = bool(
        isinstance(attacks, dict)
        and attacks.get("passed") is True
        and holder_policy_attacks_passed
        and protocol_rejections_passed
        and isinstance(valid_presentation, dict)
        and valid_presentation.get("accepted") is True
        and process_boundary_passed
    )
    decision_status = (
        "automated_security_controls_passed"
        if attacks_passed
        else "automated_attack_evidence_incomplete"
    )
    return {
        "schema": "mettle-presence-three-party-relay-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": base_url.rstrip("/"),
        "suites": list(BRONZE_SUITES),
        "automated_security_criteria": AUTOMATED_SECURITY_CRITERIA,
        "optional_timing_calibration_criteria": OPTIONAL_TIMING_CALIBRATION_CRITERIA,
        "attestation_signature_verified": True,
        "cohorts": [asdict(cohort) for cohort in cohorts],
        "summaries": summaries,
        "separation": separation,
        "attack_trials": attacks,
        "decision": {
            "status": decision_status,
            "authorization_controls_validated": attacks_passed,
            "threshold_enforcement_authorized": False,
            "human_testing_required": False,
            "measured_human_cohort_status": (
                "completed" if measured_human_present else "not_required"
            ),
            "reason": (
                "Autonomous attack trials validate holder authorization and protocol "
                "boundaries. Timing cohorts remain descriptive and do not authorize "
                "product gating."
            ),
        },
        "interpretation_limits": [
            "The holder and solver process boundary is real and the private key never crosses IPC.",
            "Holder-service and paced-relay delays are injected locally, not measured network or human latency.",
            "The process relay demonstrates solver adaptation through the public challenge surface.",
            "Key binding rejects theft and copying without the holder, but a cooperating holder service remains possible.",
            "Automated attack success validates authorization controls, not identity or product timing enforcement.",
        ],
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="METTLE_API_KEY")
    parser.add_argument(
        "--cohort",
        action="append",
        type=parse_cohort,
        dest="cohorts",
        help="Repeatable NAME:KIND:DELAY_MS:SESSIONS specification",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = args.cohorts or [
        Cohort("direct", "direct", 0, 2),
        Cohort("process_relay", "process-relay", 0, 2),
        Cohort("holder_service_250", "holder-service", 250, 2),
        Cohort("paced_relay_1000", "paced-relay", 1000, 2),
    ]
    if len({cohort.name for cohort in cohorts}) != len(cohorts):
        parser.error("cohort names must be unique")
    if not any(cohort.kind == "direct" for cohort in cohorts):
        parser.error("at least one direct cohort is required")
    if (
        any(cohort.kind == "manual-human" for cohort in cohorts)
        and not sys.stdin.isatty()
    ):
        parser.error("manual-human cohorts require an interactive terminal")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"{args.api_key_env} is unset or empty")

    samples: list[dict[str, Any]] = []
    for cohort in cohorts:
        for session_index in range(1, cohort.sessions + 1):
            if cohort.kind == "direct":
                session_samples = _run_direct_session(
                    base_url=args.base_url,
                    api_key=api_key,
                    cohort=cohort,
                    session_index=session_index,
                    timeout=args.timeout,
                )
            else:
                session_samples = _run_relay_session(
                    base_url=args.base_url,
                    api_key=api_key,
                    cohort=cohort,
                    session_index=session_index,
                    timeout=args.timeout,
                )
            samples.extend(session_samples)
    attacks = (
        None
        if args.skip_attacks
        else run_attack_trials(
            base_url=args.base_url, api_key=api_key, timeout=args.timeout
        )
    )
    report = build_report(
        base_url=validate_trial_url(args.base_url),
        cohorts=cohorts,
        samples=samples,
        attacks=attacks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summaries": report["summaries"],
                "decision": report["decision"],
                "attacks_passed": attacks is None or attacks["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
