"""Simple API key bearer authentication for METTLE standalone."""

import hashlib
import logging
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


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

    api_key = credentials.credentials
    environment = os.getenv("METTLE_ENVIRONMENT", "development").lower()
    dev_mode = (
        os.getenv("METTLE_DEV_MODE", "false").lower() == "true"
        and environment != "production"
    )
    valid_keys = [key.strip() for key in os.getenv("METTLE_API_KEYS", "").split(",") if key.strip()]
    valid = dev_mode or any(secrets.compare_digest(api_key, key) for key in valid_keys)
    if valid:
        key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        return AuthenticatedUser(user_id=f"key:{key_fingerprint}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
    )
