"""Shared rate-limiter authority for the API and nested METTLE router."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings


CREDENTIAL_STATUS_RATE_LIMIT = "60/minute"

_settings = get_settings()
_storage_options: dict[str, object] = {}
if _settings.redis_url:
    _storage_options = {
        "socket_connect_timeout": 1.0,
        "socket_timeout": 1.0,
    }
    if _settings.is_production and _settings.redis_url.startswith("rediss://"):
        _storage_options["ssl_cert_reqs"] = "required"
        _storage_options["ssl_check_hostname"] = True

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_settings.redis_url or "memory://",
    storage_options=_storage_options,
)
