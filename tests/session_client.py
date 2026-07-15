"""Test client that carries server-issued legacy session bearer tokens."""

import re
from typing import Any

from fastapi.testclient import TestClient


class SessionAwareTestClient(TestClient):
    """Preserve explicit token auth without repeating plumbing in flow tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.session_tokens: dict[str, str] = {}

    def request(self, method: str, url: Any, **kwargs: Any):
        path = str(url)
        session_id = None
        body = kwargs.get("json")
        if path.endswith("/api/session/answer") and isinstance(body, dict):
            session_id = body.get("session_id")
        else:
            match = re.search(r"/api/session/([^/?]+)", path)
            if match:
                session_id = match.group(1)

        if session_id and session_id in self.session_tokens:
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault("X-Session-Token", self.session_tokens[session_id])
            kwargs["headers"] = headers

        response = super().request(method, url, **kwargs)
        if method.upper() == "POST" and path.endswith("/api/session/start"):
            if response.status_code == 200:
                data = response.json()
                self.session_tokens[data["session_id"]] = data["session_token"]
        return response
