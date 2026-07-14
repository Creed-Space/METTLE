#!/usr/bin/env python3
"""JSON-lines Ed25519 holder worker for process-isolated Presence trials."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mettle.presence import presentation_signing_bytes, submission_signing_bytes


def _required_string(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sign(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


def main() -> int:
    private_key = Ed25519PrivateKey.generate()
    public_key_pem = (
        private_key.public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    for line in sys.stdin:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            action = request.get("action")
            if action == "public_key":
                result = {"public_key_pem": public_key_pem}
            elif action == "sign_submission":
                message = submission_signing_bytes(
                    session_id=_required_string(request, "session_id"),
                    action=_required_string(request, "submission_action"),
                    nonce=_required_string(request, "nonce"),
                    previous_transcript_hash=_required_string(
                        request, "previous_transcript_hash"
                    ),
                    payload_hash=_required_string(request, "payload_hash"),
                )
                result = {"signature": _sign(private_key, message)}
            elif action == "sign_presentation":
                message = presentation_signing_bytes(
                    challenge_id=_required_string(request, "challenge_id"),
                    nonce=_required_string(request, "nonce"),
                    audience=_required_string(request, "audience"),
                    credential_jti=_required_string(request, "credential_jti"),
                    expires_at=_required_string(request, "expires_at"),
                )
                result = {"signature": _sign(private_key, message)}
            elif action == "shutdown":
                print(
                    json.dumps({"id": request_id, "ok": True}, separators=(",", ":")),
                    flush=True,
                )
                return 0
            else:
                raise ValueError("unsupported holder action")
            response = {"id": request_id, "ok": True, **result}
        except (KeyError, TypeError, ValueError) as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
