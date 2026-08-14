#!/usr/bin/env python3
"""Promote one reviewed release commit to bound Render production services."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_render_drift import RENDER_API, _HTTPS_OPENER, load_contract


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{40}")
TAG_RE = re.compile(r"v[0-9A-Za-z][0-9A-Za-z.+-]{0,62}")
REQUIRED_RELEASE_SERVICES = ("mettle-api", "mettle-mcp")
FAILURE_STATUSES = {
    "build_failed",
    "update_failed",
    "pre_deploy_failed",
    "canceled",
    "deactivated",
}


class RenderAPIError(RuntimeError):
    """A secret-safe Render control-plane failure."""


class RenderPromotionError(RenderAPIError):
    """A failed multi-service promotion with its attempted rollback receipts."""

    def __init__(self, message: str, rollbacks: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.rollbacks = rollbacks


def _request_json(
    path: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, object | None]:
    """Call only the fixed Render HTTPS origin and reject every redirect."""
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("Render API path must stay under the fixed HTTPS origin")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "METTLE-Release-Promotion/1.0",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{RENDER_API}{path}", data=data, headers=headers, method=method
    )
    try:
        with _HTTPS_OPENER.open(request, timeout=30) as response:  # nosec B310
            final = urllib.parse.urlsplit(response.geturl())
            expected = urllib.parse.urlsplit(RENDER_API)
            if (final.scheme, final.netloc) != (expected.scheme, expected.netloc):
                raise ValueError("Render API response changed origin")
            status = int(response.status)
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RenderAPIError(
            f"Render API {method} {path.split('?', 1)[0]} returned HTTP {exc.code}"
        ) from exc
    if not body:
        return status, None
    try:
        return status, json.loads(body)
    except json.JSONDecodeError as exc:
        raise RenderAPIError("Render API returned malformed JSON") from exc


def release_targets(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Select exactly the two production services held behind tag promotion."""
    targets: list[dict[str, str]] = []
    for name, service in contract["services"].items():
        binding = service["binding"]
        if binding.get("promote_on_release") is not True:
            continue
        if service["blueprint"].get("autoDeploy") is not False:
            raise ValueError(f"release-managed service must disable autoDeploy: {name}")
        service_id = binding.get("service_id")
        if not isinstance(service_id, str) or not service_id.startswith("srv-"):
            raise ValueError(f"release-managed service has an invalid ID: {name}")
        targets.append({"name": name, "service_id": service_id})
    by_name = {target["name"]: target for target in targets}
    if set(by_name) != set(REQUIRED_RELEASE_SERVICES):
        raise ValueError("release promotion must bind exactly the API and MCP services")
    return [by_name[name] for name in REQUIRED_RELEASE_SERVICES]


def _deploys(payload: object | None) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise RenderAPIError("Render deploy list has an unexpected shape")
    deployments: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RenderAPIError("Render deploy list contains an invalid item")
        deployment = item.get("deploy", item)
        if not isinstance(deployment, dict):
            raise RenderAPIError("Render deploy list contains an invalid deploy")
        deployments.append(deployment)
    return deployments


def _list_deploys(service_id: str, token: str) -> list[dict[str, Any]]:
    _, payload = _request_json(f"/services/{service_id}/deploys?limit=20", token)
    return _deploys(payload)


def _matching_deploy(
    deployments: list[dict[str, Any]], source_revision: str
) -> dict[str, Any] | None:
    matches = [
        deployment
        for deployment in deployments
        if isinstance(deployment.get("commit"), dict)
        and deployment["commit"].get("id") == source_revision
        and deployment.get("status") != "deactivated"
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: str(item.get("createdAt", "")))


def _wait_for_queued_deploy(
    service_id: str,
    source_revision: str,
    token: str,
    *,
    deadline: float,
    poll_seconds: float,
) -> dict[str, Any]:
    while time.monotonic() < deadline:
        match = _matching_deploy(_list_deploys(service_id, token), source_revision)
        if match is not None:
            return match
        time.sleep(poll_seconds)
    raise TimeoutError("queued Render deploy did not become observable before timeout")


def _wait_until_live(
    service_id: str,
    deployment: dict[str, Any],
    source_revision: str,
    token: str,
    *,
    deadline: float,
    poll_seconds: float,
    allowed_triggers: set[object] | None = None,
) -> dict[str, Any]:
    deploy_id = deployment.get("id")
    if not isinstance(deploy_id, str) or not deploy_id.startswith("dep-"):
        raise RenderAPIError("Render deploy response has no valid deploy ID")
    current = deployment
    while True:
        status = current.get("status")
        if status == "live":
            commit = current.get("commit")
            if not isinstance(commit, dict) or commit.get("id") != source_revision:
                raise RenderAPIError("live Render deploy does not match release commit")
            permitted = allowed_triggers or {None, "api"}
            if current.get("trigger") not in permitted:
                raise RenderAPIError("live Render deploy has an unexpected trigger")
            return current
        if status in FAILURE_STATUSES:
            raise RenderAPIError(f"Render deploy reached terminal status {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Render deploy did not become live before timeout")
        time.sleep(poll_seconds)
        _, payload = _request_json(f"/services/{service_id}/deploys/{deploy_id}", token)
        if not isinstance(payload, dict):
            raise RenderAPIError("Render deploy response has an unexpected shape")
        current = payload


def promote_service(
    target: dict[str, str],
    source_revision: str,
    token: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    on_prepared: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Deploy one exact commit and return a nonsecret provider receipt."""
    service_id = target["service_id"]
    previous = next(
        (
            deployment
            for deployment in _list_deploys(service_id, token)
            if deployment.get("status") == "live"
        ),
        None,
    )
    if (
        not isinstance(previous, dict)
        or not isinstance(previous.get("id"), str)
        or not str(previous["id"]).startswith("dep-")
        or not isinstance(previous.get("commit"), dict)
        or not isinstance(previous["commit"].get("id"), str)
        or SOURCE_REVISION_RE.fullmatch(previous["commit"]["id"]) is None
    ):
        raise RenderAPIError("Render service has no valid live rollback target")
    prepared: dict[str, object] = {
        "name": target["name"],
        "service_id": service_id,
        "status": "attempting",
        "commit_id": source_revision,
        "previous_live_deploy_id": previous["id"],
        "previous_live_commit_id": previous["commit"]["id"],
    }
    if on_prepared is not None:
        on_prepared(prepared)
    deadline = time.monotonic() + timeout_seconds
    status_code, payload = _request_json(
        f"/services/{service_id}/deploys",
        token,
        method="POST",
        payload={"commitId": source_revision, "clearCache": "do_not_clear"},
    )
    if status_code not in {201, 202}:
        raise RenderAPIError(f"Render deploy trigger returned HTTP {status_code}")
    if isinstance(payload, dict):
        deployment = payload
    elif status_code == 202:
        deployment = _wait_for_queued_deploy(
            service_id,
            source_revision,
            token,
            deadline=deadline,
            poll_seconds=poll_seconds,
        )
    else:
        raise RenderAPIError("Render deploy trigger returned an unexpected shape")
    live = _wait_until_live(
        service_id,
        deployment,
        source_revision,
        token,
        deadline=deadline,
        poll_seconds=poll_seconds,
    )
    return {
        "name": target["name"],
        "service_id": service_id,
        "deploy_id": live["id"],
        "status": "live",
        "commit_id": source_revision,
        "trigger": live.get("trigger"),
        "started_at": live.get("startedAt"),
        "finished_at": live.get("finishedAt"),
        "previous_live_deploy_id": (
            previous.get("id") if isinstance(previous, dict) else None
        ),
        "previous_live_commit_id": previous["commit"]["id"],
    }


def rollback_service(
    promoted: dict[str, object],
    token: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, object]:
    """Restore one previously live deploy after a later service fails."""
    service_id = promoted.get("service_id")
    previous_deploy_id = promoted.get("previous_live_deploy_id")
    previous_commit_id = promoted.get("previous_live_commit_id")
    if (
        not isinstance(service_id, str)
        or not service_id.startswith("srv-")
        or not isinstance(previous_deploy_id, str)
        or not previous_deploy_id.startswith("dep-")
        or not isinstance(previous_commit_id, str)
        or SOURCE_REVISION_RE.fullmatch(previous_commit_id) is None
    ):
        raise RenderAPIError("promotion receipt has no valid rollback target")
    status_code, payload = _request_json(
        f"/services/{service_id}/rollback",
        token,
        method="POST",
        payload={"deployId": previous_deploy_id},
    )
    if status_code != 201 or not isinstance(payload, dict):
        raise RenderAPIError("Render rollback trigger returned an unexpected response")
    live = _wait_until_live(
        service_id,
        payload,
        previous_commit_id,
        token,
        deadline=time.monotonic() + timeout_seconds,
        poll_seconds=poll_seconds,
        allowed_triggers={None, "api", "rollback"},
    )
    return {
        "name": promoted["name"],
        "service_id": service_id,
        "rollback_deploy_id": live["id"],
        "restored_commit_id": previous_commit_id,
        "status": "live",
    }


def promote_release(
    contract: dict[str, Any],
    source_revision: str,
    tag: str,
    token: str,
    *,
    timeout_seconds: float = 1800,
    poll_seconds: float = 10,
) -> dict[str, object]:
    if SOURCE_REVISION_RE.fullmatch(source_revision) is None:
        raise ValueError("source revision must be a lowercase 40-character Git SHA")
    if TAG_RE.fullmatch(tag) is None:
        raise ValueError("release tag has an invalid shape")
    if not token:
        raise ValueError("Render API token input is empty")
    services: list[dict[str, object]] = []
    try:
        for target in release_targets(contract):
            prepared_index = len(services)
            completed = promote_service(
                target,
                source_revision,
                token,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                on_prepared=services.append,
            )
            if len(services) != prepared_index + 1:
                raise RenderAPIError("Render promotion bookkeeping is inconsistent")
            services[prepared_index] = completed
    except (RenderAPIError, TimeoutError) as promotion_error:
        rollbacks: list[dict[str, object]] = []
        rollback_failed = False
        for promoted in reversed(services):
            try:
                rollbacks.append(
                    rollback_service(
                        promoted,
                        token,
                        timeout_seconds=timeout_seconds,
                        poll_seconds=poll_seconds,
                    )
                )
            except (RenderAPIError, TimeoutError) as rollback_error:
                rollback_failed = True
                rollbacks.append(
                    {
                        "name": promoted["name"],
                        "service_id": promoted["service_id"],
                        "status": "error",
                        "error_type": type(rollback_error).__name__,
                    }
                )
        if services:
            outcome = "rollback incomplete" if rollback_failed else "services restored"
            raise RenderPromotionError(
                f"Render release promotion failed; {outcome}", rollbacks
            ) from promotion_error
        raise
    return {
        "schema_version": "1.0",
        "promoted_at": datetime.now(UTC).isoformat(),
        "result": "live",
        "tag": tag,
        "source_revision": source_revision,
        "blueprint_sha256": contract["blueprint_sha256"],
        "deployment_sha256": contract["deployment_sha256"],
        "services": services,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", type=Path, default=ROOT / "render.yaml")
    parser.add_argument(
        "--deployment", type=Path, default=ROOT / "deploy/render-production.json"
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.token_stdin:
        parser.error("pass the Render API token through --token-stdin")
    token = sys.stdin.read().strip()
    exit_status = 0
    try:
        contract = load_contract(args.blueprint, args.deployment)
        receipt = promote_release(
            contract,
            args.source_revision,
            args.tag,
            token,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except (
        KeyError,
        OSError,
        RenderAPIError,
        TimeoutError,
        ValueError,
        urllib.error.URLError,
    ) as error:
        receipt = {
            "schema_version": "1.0",
            "promoted_at": datetime.now(UTC).isoformat(),
            "result": "error",
            "tag": args.tag,
            "source_revision": args.source_revision,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if isinstance(error, RenderPromotionError):
            receipt["rollbacks"] = error.rollbacks
        exit_status = 1
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
