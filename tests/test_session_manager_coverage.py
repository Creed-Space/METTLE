"""Tests for mettle/session_manager.py coverage gaps — lines 101, 105.

Line 101: intent-provenance suite with vcp_token passed through
Line 105: Unknown suite name raises ValueError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mettle.session_manager import SessionManager


@pytest.fixture
def mock_redis():
    """Create a mock Redis client with realistic async return values."""
    redis = AsyncMock()
    # Rate limiting: scard returns 0 active sessions, get returns None (no hourly count)
    redis.scard = AsyncMock(return_value=0)
    redis.get = AsyncMock(return_value=None)

    # Pipeline: returns a mock that has sync method calls and async execute
    pipe = MagicMock()
    pipe.setex = MagicMock(return_value=pipe)
    pipe.sadd = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.incr = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


class TestCreateSessionCoverage:
    @pytest.mark.asyncio
    async def test_unknown_suite_raises_value_error(self, mock_redis):
        """Unknown suite name raises ValueError (line 105)."""
        mgr = SessionManager(mock_redis)

        with pytest.raises(ValueError, match="Unknown suite|Invalid suite"):
            await mgr.create_session(
                user_id="test-user",
                suites=["nonexistent-suite"],
                difficulty="standard",
            )

    @pytest.mark.asyncio
    async def test_intent_provenance_with_vcp_token(self, mock_redis):
        """intent-provenance suite with vcp_token triggers VCP-enhanced path (line 101)."""
        mgr = SessionManager(mock_redis)

        vcp_token = (
            "VCP:3.1:agent-42\n"
            "C:professional.safe@2.0.0\n"
            "P:advisor:4\n"
        )

        session_id, client_challenges, session_meta = await mgr.create_session(
            user_id="test-user",
            suites=["intent-provenance"],
            difficulty="standard",
            vcp_token=vcp_token,
        )

        assert session_id is not None
        assert "intent-provenance" in client_challenges
        # VCP-enhanced challenges should include vcp_token_verification
        challenges = client_challenges["intent-provenance"].get("challenges", {})
        assert "vcp_token_verification" in challenges
