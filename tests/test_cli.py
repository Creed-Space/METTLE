"""Security and behavior tests for the METTLE CLI evidence flow."""

import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from mettle import cli
from mettle.models import Challenge, ChallengeType


def _receipt(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("receipt_type"):
            return value
    raise AssertionError("No evidence receipt found")


def _known_challenge(index: int = 1) -> Challenge:
    return Challenge(
        id=f"mtl_test_{index:04d}",
        type=ChallengeType.SPEED_MATH,
        prompt="Calculate: 40 + 2",
        data={"expected_answer": 42, "a": 40, "b": 2, "op": "+"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        time_limit_ms=60000,
    )


def _complete_known_battery() -> list[Challenge]:
    return [_known_challenge(index) for index in range(1, 4)]


def test_auto_solver_flag_is_removed():
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", "--auto"])
    assert exc.value.code == 2


def test_notarization_flag_is_removed():
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", "--notarize"])
    assert exc.value.code == 2


def test_interactive_pass_emits_unsigned_noncredential(capsys, monkeypatch):
    monkeypatch.setattr(
        cli, "generate_challenge_set", lambda difficulty: _complete_known_battery()
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("42\n42\n42\n"))

    assert cli.main(["verify", "--json"]) == 0
    receipt = _receipt(capsys.readouterr().out)
    assert receipt["screening_passed"] is True
    assert receipt["verified"] is True
    assert receipt["tier"] == "bronze"
    assert receipt["credential_eligible"] is False
    assert receipt["assurance"] == "mettle_local_behavioral_verification"
    assert receipt["signature"] is None
    assert cli.verify_credential(receipt) is False


def test_interactive_wrong_answer_fails_screening(capsys, monkeypatch):
    monkeypatch.setattr(
        cli, "generate_challenge_set", lambda difficulty: _complete_known_battery()
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("99\n99\n99\n"))

    assert cli.main(["verify", "--json"]) == 1
    receipt = _receipt(capsys.readouterr().out)
    assert receipt["screening_passed"] is False
    assert receipt["verified"] is False


def test_bad_suite_name_exits_2(capsys):
    assert cli.main(["verify", "--suite", "nope"]) == 2
    assert "Unknown suite" in capsys.readouterr().err


def test_suites_command_lists_registry(capsys):
    assert cli.main(["suites"]) == 0
    out = capsys.readouterr().out
    assert "adversarial" in out
    assert "governance" in out


@pytest.mark.parametrize("payload", ["[]\n", "null\n", "true\n", "42\n", '"answer"\n'])
def test_suite_answer_json_root_must_be_an_object(payload, capsys, monkeypatch):
    monkeypatch.setattr(cli.ChallengeAdapter, "generate_adversarial", lambda: ({}, {}))
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    assert cli.main(["verify", "--suite", "adversarial", "--json"]) == 2
    assert "must be an object" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "message"),
    [("{broken\n", "Invalid JSON answer"), ("\n", "must be a non-empty object")],
)
def test_suite_answer_requires_nonempty_valid_json(
    payload, message, capsys, monkeypatch
):
    monkeypatch.setattr(cli.ChallengeAdapter, "generate_adversarial", lambda: ({}, {}))
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    assert cli.main(["verify", "--suite", "adversarial", "--json"]) == 2
    assert message in capsys.readouterr().err


def test_suite_answer_rejects_oversized_input_before_evaluation(capsys, monkeypatch):
    monkeypatch.setattr(cli.ChallengeAdapter, "generate_adversarial", lambda: ({}, {}))
    monkeypatch.setattr(
        cli.ChallengeAdapter,
        "evaluate_single_shot",
        lambda *_args: pytest.fail("oversized input reached evaluation"),
    )
    payload = '{"answer":"' + ("x" * cli._MAX_SUITE_ANSWER_BYTES) + '"}\n'
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    assert cli.main(["verify", "--suite", "adversarial", "--json"]) == 2
    assert "byte limit" in capsys.readouterr().err


def test_quick_answer_rejects_oversized_input(capsys, monkeypatch):
    monkeypatch.setattr(
        cli, "generate_challenge_set", lambda difficulty: _complete_known_battery()
    )
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("x" * (cli._MAX_QUICK_ANSWER_BYTES + 1) + "\n")
    )

    assert cli.main(["verify", "--json"]) == 2
    assert "byte limit" in capsys.readouterr().err
