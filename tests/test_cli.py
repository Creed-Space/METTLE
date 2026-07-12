"""End-to-end tests for the `mettle` CLI (mettle.cli)."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from mettle import cli
from mettle.models import Challenge, ChallengeType


@pytest.fixture(autouse=True)
def isolated_key_home(tmp_path, monkeypatch):
    """Point the CLI signing key at a temp dir so we never touch ~/.mettle."""
    monkeypatch.setenv("METTLE_HOME", str(tmp_path / "mettle_home"))
    yield


def _credential_lines(captured: str) -> list[dict]:
    """Parse all JSON lines from captured stdout."""
    lines = []
    for raw in captured.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.append(json.loads(raw))
    return lines


def _find_credential(captured: str) -> dict:
    for obj in _credential_lines(captured):
        if obj.get("credential_type") == "mettle-self-signed":
            return obj
    raise AssertionError("no credential emitted")


def test_auto_json_basic_signature_verifies(capsys):
    """`mettle verify --auto --json` yields a valid, signed credential with a tier."""
    exit_code = cli.main(["verify", "--auto", "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    # --json emits ONLY the credential line on stdout.
    credential = _find_credential(captured.out)

    assert credential["tier"] in {"bronze", "silver", "gold", "platinum", "none"}
    assert credential["tier"] == "bronze"  # basic + verified
    assert credential["verified"] is True
    assert "public_key_pem" in credential
    assert credential["signature"].startswith("ed25519:")

    # Signature verifies with the emitted public key (independent check).
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    claims = {
        k: v for k, v in credential.items() if k not in ("signature", "public_key_pem")
    }
    signed_bytes = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    sig = credential["signature"].split("ed25519:", 1)[1]

    import base64

    public_key = load_pem_public_key(credential["public_key_pem"].encode("ascii"))
    assert isinstance(public_key, Ed25519PublicKey)
    # Raises if invalid.
    public_key.verify(base64.b64decode(sig), signed_bytes)

    # And the module helper agrees.
    assert cli.verify_credential(credential) is True


def test_tamper_breaks_signature(capsys):
    cli.main(["verify", "--auto", "--json"])
    credential = _find_credential(capsys.readouterr().out)
    credential["tier"] = "platinum"  # tamper
    assert cli.verify_credential(credential) is False


def test_bad_suite_name_exits_2(capsys):
    exit_code = cli.main(["verify", "--suite", "does-not-exist"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Unknown suite" in err


def test_suite_auto_native_verifies(capsys):
    exit_code = cli.main(["verify", "--suite", "native", "--auto", "--json"])
    assert exit_code == 0
    credential = _find_credential(capsys.readouterr().out)
    assert credential["verified"] is True
    assert credential["suites_passed"] == ["native"]
    assert cli.verify_credential(credential) is True


def test_full_auto_reaches_higher_tier(capsys):
    exit_code = cli.main(["verify", "--full", "--auto", "--json"])
    assert exit_code == 0
    credential = _find_credential(capsys.readouterr().out)
    assert credential["verified"] is True
    assert credential["tier"] == "silver"


def test_notarize_unavailable_exits_2(capsys):
    exit_code = cli.main(["verify", "--auto", "--notarize"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "not available" in err.lower()


def test_interactive_piped_answers(capsys, monkeypatch):
    """Interactive mode reads one answer line per challenge from stdin."""
    known = Challenge(
        id="mtl_test_0001",
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate: 40 + 2",
        data={"expected_answer": 42, "a": 40, "b": 2, "op": "+"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=60000,
    )
    monkeypatch.setattr(cli, "generate_challenge_set", lambda difficulty: [known])
    monkeypatch.setattr("sys.stdin", io.StringIO("42\n"))

    exit_code = cli.main(["verify"])
    assert exit_code == 0

    captured = capsys.readouterr()
    credential = _find_credential(captured.out)
    assert credential["verified"] is True
    assert credential["mode"] == "interactive"
    assert "speed_math" in credential["suites_passed"]


def test_interactive_wrong_answer_fails(capsys, monkeypatch):
    known = Challenge(
        id="mtl_test_0002",
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate: 40 + 2",
        data={"expected_answer": 42, "a": 40, "b": 2, "op": "+"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=60000,
    )
    monkeypatch.setattr(cli, "generate_challenge_set", lambda difficulty: [known])
    monkeypatch.setattr("sys.stdin", io.StringIO("99\n"))

    exit_code = cli.main(["verify"])
    assert exit_code == 1  # ran cleanly, failed verification
    credential = _find_credential(capsys.readouterr().out)
    assert credential["verified"] is False


def test_suites_command_lists_registry(capsys):
    exit_code = cli.main(["suites"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "adversarial" in out
    assert "governance" in out
