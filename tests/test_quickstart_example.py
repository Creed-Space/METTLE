"""Execute the published admission example against real local API routes."""

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import main


class QuickStartParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.code = ""

    def handle_starttag(self, tag, attrs):
        if tag == "code" and dict(attrs).get("id") == "quickstart-python":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "code":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.code += data


@pytest.fixture
def example(monkeypatch):
    parser = QuickStartParser()
    parser.feed(Path("static/docs.html").read_text())
    assert parser.code
    # Plain TestClient: no helper silently adds the required session header.
    transport = TestClient(main.app)

    class LocalClient:
        def __enter__(self):
            return transport

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(httpx, "Client", lambda **_: LocalClient())
    namespace: dict[str, Any] = {}
    exec(compile(parser.code, "quickstart-python", "exec"), namespace)  # noqa: S102
    yield namespace
    transport.close()


@pytest.mark.parametrize("answer, passes", [("42", True), ("wrong", False)])
def test_quickstart_admission(
    example, monkeypatch, sample_speed_math_challenge, answer, passes
):
    # Known test fixtures, never a solver for real challenges.
    monkeypatch.setattr(
        main,
        "generate_challenge_set",
        lambda _: [
            sample_speed_math_challenge.model_copy(update={"id": f"mtl_{i:024x}"})
            for i in range(3)
        ],
    )
    badge = example["earn_badge"](lambda challenge: answer)
    assert bool(badge) is passes
    assert example["check_badge"](badge) is passes
    assert example["check_badge"]("invalid-badge") is False
    if badge:
        payload = main._verify_badge_token(badge).payload
        assert payload is not None
        monkeypatch.setitem(main.revoked_badges, payload["jti"], 0.0)
        assert example["check_badge"](badge) is False


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("offline"), ValueError("bad JSON")]
)
def test_quickstart_denies_when_verifier_unavailable(example, monkeypatch, failure):
    def unavailable(**_):
        raise failure

    monkeypatch.setattr(httpx, "Client", unavailable)
    assert example["check_badge"]("test-badge") is False
