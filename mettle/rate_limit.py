"""Shared rate-limiter authority for the API and nested METTLE router."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings


CREDENTIAL_STATUS_RATE_LIMIT = "60/minute"

_settings = get_settings()
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_settings.redis_url or "memory://",
)
