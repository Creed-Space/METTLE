#!/usr/bin/env python3
"""JSON-lines Ed25519 holder worker for process-isolated Presence trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mettle.holder import EphemeralEd25519Signer, HolderPolicy, PresenceHolder


def _required_string(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def main() -> int:
    signer = EphemeralEd25519Signer()
    holder: PresenceHolder | None = None
    for line in sys.stdin:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            action = request.get("action")
            if action == "public_key":
                result = {"public_key_pem": signer.public_key_pem}
            elif action == "configure":
                if holder is not None:
                    raise ValueError("holder policy is already configured")
                issuer = _required_string(request, "issuer")
                issuer_public_key_pem = _required_string(
                    request, "issuer_public_key_pem"
                )
                audiences = request.get("allowed_audiences")
                if not isinstance(audiences, list) or any(
                    not isinstance(value, str) for value in audiences
                ):
                    raise ValueError("allowed_audiences must be a string list")
                holder = PresenceHolder(
                    signer,
                    HolderPolicy(
                        issuer_public_keys={issuer: issuer_public_key_pem},
                        allowed_audiences=frozenset(audiences),
                        max_active_sessions=int(request.get("max_active_sessions", 1)),
                        max_actions_per_session=int(
                            request.get("max_actions_per_session", 16)
                        ),
                        max_presentations_per_credential=int(
                            request.get("max_presentations_per_credential", 32)
                        ),
                        max_presentation_ttl_seconds=int(
                            request.get("max_presentation_ttl_seconds", 600)
                        ),
                    ),
                )
                result = holder.status()
            elif holder is None:
                raise ValueError("holder policy is not configured")
            elif action == "authorize_session":
                presence = request.get("presence")
                if not isinstance(presence, dict):
                    raise ValueError("presence must be an object")
                holder.authorize_session(
                    issuer=_required_string(request, "issuer"),
                    session_id=_required_string(request, "session_id"),
                    presence=presence,
                )
                result = holder.status()
            elif action == "sign_submission":
                result = {
                    "signature": holder.sign_submission(
                        session_id=_required_string(request, "session_id"),
                        action=_required_string(request, "submission_action"),
                        nonce=_required_string(request, "nonce"),
                        previous_transcript_hash=_required_string(
                            request, "previous_transcript_hash"
                        ),
                        payload_hash=_required_string(request, "payload_hash"),
                    )
                }
            elif action == "commit_submission":
                presence = request.get("presence")
                if not isinstance(presence, dict):
                    raise ValueError("presence must be an object")
                holder.commit_submission(
                    session_id=_required_string(request, "session_id"),
                    presence=presence,
                )
                result = holder.status()
            elif action == "register_credential":
                attestation = request.get("attestation")
                if not isinstance(attestation, dict):
                    raise ValueError("attestation must be an object")
                status_receipt = request.get("status_receipt")
                if not isinstance(status_receipt, dict):
                    raise ValueError("status_receipt must be an object")
                credential_jti = holder.register_credential(
                    issuer=_required_string(request, "issuer"),
                    attestation=attestation,
                    status_receipt=status_receipt,
                )
                result = {"credential_jti": credential_jti, **holder.status()}
            elif action == "sign_presentation":
                result = {
                    "signature": holder.sign_presentation(
                        challenge_id=_required_string(request, "challenge_id"),
                        nonce=_required_string(request, "nonce"),
                        audience=_required_string(request, "audience"),
                        credential_jti=_required_string(request, "credential_jti"),
                        expires_at=_required_string(request, "expires_at"),
                    )
                }
            elif action == "status":
                result = holder.status()
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
