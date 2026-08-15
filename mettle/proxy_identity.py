"""Restore a Cloudflare visitor address only from an authenticated proxy hop.

Uvicorn first reduces Render's forwarded chain to the rightmost untrusted hop.
For the public METTLE hostname that hop is a Cloudflare edge address, not the
visitor. Cloudflare supplies the visitor separately in ``CF-Connecting-IP``.
Trust that single-value header only when Uvicorn has already established that
the immediate public hop belongs to Cloudflare.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence

from starlette.types import ASGIApp, Receive, Scope, Send


# Snapshot of Cloudflare's authoritative IPv4 and IPv6 lists on 2026-08-15.
# Sources: https://www.cloudflare.com/ips-v4 and /ips-v6
_CLOUDFLARE_CIDRS: tuple[str, ...] = (
    "173.245.48.0/20",
    "103.21.244.0/22",
    "103.22.200.0/22",
    "103.31.4.0/22",
    "141.101.64.0/18",
    "108.162.192.0/18",
    "190.93.240.0/20",
    "188.114.96.0/20",
    "197.234.240.0/22",
    "198.41.128.0/17",
    "162.158.0.0/15",
    "104.16.0.0/13",
    "104.24.0.0/14",
    "172.64.0.0/13",
    "131.0.72.0/22",
    "2400:cb00::/32",
    "2606:4700::/32",
    "2803:f800::/32",
    "2405:b500::/32",
    "2405:8100::/32",
    "2a06:98c0::/29",
    "2c0f:f248::/32",
)
CLOUDFLARE_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in _CLOUDFLARE_CIDRS)


def _is_cloudflare_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(address in network for network in CLOUDFLARE_NETWORKS)


def _single_connecting_ip(headers: Sequence[tuple[bytes, bytes]]) -> str | None:
    values = [value for name, value in headers if name.lower() == b"cf-connecting-ip"]
    if len(values) != 1:
        return None
    try:
        candidate = values[0].decode("ascii").strip()
        return str(ipaddress.ip_address(candidate))
    except (UnicodeDecodeError, ValueError):
        return None


class CloudflareClientIPMiddleware:
    """Expose Cloudflare's authenticated visitor address as the ASGI client."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        if client and _is_cloudflare_address(client[0]):
            visitor = _single_connecting_ip(scope.get("headers", []))
            if visitor is not None:
                scope = dict(scope)
                scope["client"] = (visitor, 0)
        await self.app(scope, receive, send)
