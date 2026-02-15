"""Simple API key bearer authentication for METTLE standalone."""

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)
security = HTTPBearer()


class AuthenticatedUser(BaseModel):
    user_id: str


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    api_key = credentials.credentials
    dev_mode = os.getenv("METTLE_DEV_MODE", "false").lower() == "true"
    valid_keys = os.getenv("METTLE_API_KEYS", "").split(",")
    if dev_mode or api_key in valid_keys:
        return AuthenticatedUser(user_id=f"key:{api_key[:8]}...")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
    )
