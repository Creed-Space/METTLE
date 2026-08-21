"""Simple API key bearer authentication for METTLE standalone."""

import hashlib
import logging
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import RuntimeEnvironment, normalize_runtime_environment

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)
_DEV_BYPASS_ENVIRONMENTS = {
    RuntimeEnvironment.LOCAL,
    RuntimeEnvironment.DEVELOPMENT,
    RuntimeEnvironment.TEST,
}
_MAX_API_KEY_CHARS = 512


class AuthenticatedUser(BaseModel):
    user_id: str


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthenticatedUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    if credentials.scheme.casefold() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )

    api_key = credentials.credentials
    if not api_key or len(api_key) > _MAX_API_KEY_CHARS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    try:
        environment = normalize_runtime_environment(
            os.getenv("METTLE_ENVIRONMENT", "development")
        )
    except ValueError:
        # A misspelled or invented environment must never enable a bypass.
        environment = None
    dev_mode = (
        os.getenv("METTLE_DEV_MODE", "false").strip().casefold() == "true"
        and environment in _DEV_BYPASS_ENVIRONMENTS
    )
    valid_keys = [
        key.strip()
        for key in os.getenv("METTLE_API_KEYS", "").split(",")
        if key.strip() and len(key.strip()) <= _MAX_API_KEY_CHARS
    ]
    valid = dev_mode or any(secrets.compare_digest(api_key, key) for key in valid_keys)
    if valid:
        key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        return AuthenticatedUser(user_id=f"key:{key_fingerprint}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
    )
