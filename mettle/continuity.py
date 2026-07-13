"""Transcript-bound microchallenges for live METTLE Presence sessions."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

CONTINUITY_PROTOCOL = "mettle-continuity-v1"
CONTINUITY_CHALLENGE_KEY = "_mettle_continuity"
CONTINUITY_ANSWER_KEY = "_mettle_continuity"
_MASK_32 = (1 << 32) - 1
_STEP_COUNT = 8
_OPERATIONS = ("xor", "add", "multiply", "rotate_left")


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def new_continuity_state() -> dict[str, str]:
    """Create the private state used to derive one session's challenge family."""
    return {
        "continuity_protocol": CONTINUITY_PROTOCOL,
        "continuity_secret": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
    }


def _rotate_left(value: int, amount: int) -> int:
    amount %= 32
    return ((value << amount) | (value >> (32 - amount))) & _MASK_32


def solve_continuity_challenge(challenge: dict[str, Any]) -> int:
    """Solve a client-visible continuity microchallenge deterministically."""
    if challenge.get("protocol") != CONTINUITY_PROTOCOL:
        raise ValueError("Unsupported METTLE continuity challenge protocol")
    value = challenge.get("start")
    steps = challenge.get("steps")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _MASK_32
        or not isinstance(steps, list)
        or len(steps) != _STEP_COUNT
    ):
        raise ValueError("Malformed METTLE continuity challenge")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Malformed METTLE continuity step")
        operation = step.get("op")
        operand = step.get("operand")
        if (
            operation not in _OPERATIONS
            or not isinstance(operand, int)
            or isinstance(operand, bool)
            or operand < 0
            or operand > _MASK_32
        ):
            raise ValueError("Malformed METTLE continuity step")
        if operation == "xor":
            value ^= operand
        elif operation == "add":
            value = (value + operand) & _MASK_32
        elif operation == "multiply":
            value = (value * (operand | 1)) & _MASK_32
        else:
            value = _rotate_left(value, (operand % 31) + 1)
    return value


def _continuity_secret(presence: dict[str, Any]) -> bytes:
    encoded = presence.get("continuity_secret")
    if not isinstance(encoded, str):
        raise ValueError("Presence continuity secret is unavailable")
    try:
        secret = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Presence continuity secret is corrupt") from exc
    if len(secret) != 32:
        raise ValueError("Presence continuity secret is corrupt")
    return secret


def issue_continuity_challenge(
    presence: dict[str, Any], action: str
) -> dict[str, Any] | None:
    """Issue a microchallenge derived from the current transcript boundary."""
    if presence.get("continuity_protocol") is None:
        return None
    if presence.get("continuity_protocol") != CONTINUITY_PROTOCOL:
        raise ValueError("Unsupported Presence continuity protocol")
    if presence.get("current_continuity") is not None:
        raise RuntimeError("A Presence continuity challenge is already active")

    material = _canonical_bytes(
        {
            "protocol": CONTINUITY_PROTOCOL,
            "action": action,
            "transcript_hash": presence["transcript_hash"],
            "sequence": int(presence.get("sequence", 0)) + 1,
        }
    )
    secret = _continuity_secret(presence)
    stream = b"".join(
        hmac.new(secret, material + bytes([counter]), hashlib.sha256).digest()
        for counter in range(2)
    )
    start = int.from_bytes(stream[:4], "big")
    steps = []
    offset = 4
    for _ in range(_STEP_COUNT):
        operation = _OPERATIONS[stream[offset] % len(_OPERATIONS)]
        operand = int.from_bytes(stream[offset + 1 : offset + 5], "big")
        steps.append({"op": operation, "operand": operand})
        offset += 5
    challenge_id = hashlib.sha256(material + stream).hexdigest()[:32]
    challenge = {
        "protocol": CONTINUITY_PROTOCOL,
        "challenge_id": challenge_id,
        "start": start,
        "steps": steps,
    }
    presence["current_continuity"] = {
        "protocol": CONTINUITY_PROTOCOL,
        "challenge_id": challenge_id,
        "action": action,
        "expected": solve_continuity_challenge(challenge),
        "issued_at_unix_ms": int(time.time() * 1000),
    }
    return challenge


def attach_continuity_challenge(
    presence: dict[str, Any], action: str, client_payload: dict[str, Any]
) -> dict[str, Any]:
    """Copy a suite payload and attach its just-in-time continuity challenge."""
    issued = copy.deepcopy(client_payload)
    challenge = issue_continuity_challenge(presence, action)
    if challenge is not None:
        issued[CONTINUITY_CHALLENGE_KEY] = challenge
    return issued


def verify_continuity_answer(
    presence: dict[str, Any] | None,
    action: str,
    answers: dict[str, Any],
) -> None:
    """Fail closed when a new Presence session lacks its current interlock answer."""
    if presence is None or presence.get("continuity_protocol") is None:
        return
    current = presence.get("current_continuity")
    answer = answers.get(CONTINUITY_ANSWER_KEY)
    if not isinstance(current, dict) or current.get("action") != action:
        raise ValueError("Presence continuity challenge is unavailable")
    if not isinstance(answer, dict):
        raise ValueError("Presence continuity answer is required")
    if not hmac.compare_digest(
        str(answer.get("challenge_id", "")), str(current.get("challenge_id", ""))
    ):
        raise ValueError("Presence continuity challenge does not match current action")
    computed = answer.get("computed")
    if not isinstance(computed, int) or isinstance(computed, bool):
        raise ValueError("Presence continuity answer is invalid")
    if computed != current.get("expected"):
        raise ValueError("Presence continuity answer is incorrect")


def consume_continuity_evidence(presence: dict[str, Any]) -> dict[str, Any]:
    """Consume the verified microchallenge and return signed receipt fields."""
    if presence.get("continuity_protocol") is None:
        return {}
    current = presence.pop("current_continuity", None)
    if not isinstance(current, dict):
        raise RuntimeError("Verified Presence continuity evidence is missing")
    return {
        "challenge_family": current["protocol"],
        "challenge_id": current["challenge_id"],
    }


def retire_continuity_secret(presence: dict[str, Any]) -> None:
    """Remove derivation material once no future session action can be issued."""
    if presence.get("current_continuity") is not None:
        raise RuntimeError("Cannot retire an active Presence continuity challenge")
    presence.pop("continuity_secret", None)
