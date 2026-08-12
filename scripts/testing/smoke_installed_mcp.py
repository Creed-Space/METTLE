"""Smoke the installed wheel's stdio MCP tool surface."""

from __future__ import annotations

import asyncio
import os
import sys


EXPECTED_TOOLS = {
    "mettle_start_session",
    "mettle_answer_challenge",
    "mettle_get_result",
    "mettle_list_suites",
    "mettle_start_v2_session",
    "mettle_verify_suite",
    "mettle_get_v2_result",
}


async def smoke() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mettle.mcp_server"],
        env={**os.environ, "METTLE_API_URL": "https://mettle.sh/api"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    names = {tool.name for tool in result.tools}
    if names != EXPECTED_TOOLS:
        raise RuntimeError(
            f"installed MCP tools differ: expected {sorted(EXPECTED_TOOLS)}, "
            f"observed {sorted(names)}"
        )
    if "mettle_auto_verify" in names:
        raise RuntimeError("forbidden automatic solver is present")
    print("installed MCP smoke passed: 7 approved tools")


if __name__ == "__main__":
    asyncio.run(smoke())
