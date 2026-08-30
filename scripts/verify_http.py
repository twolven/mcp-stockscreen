"""Verify a running container's health route and MCP tool discovery."""

import asyncio
import sys
import time
import urllib.error
import urllib.request

from fastmcp import Client

EXPECTED_TOOLS = {
    "run_stock_screen",
    "get_stock_news",
    "manage_watchlist",
    "get_screening_result",
}


def wait_until_healthy(url: str) -> None:
    health_url = f"{url.rsplit('/', 1)[0]}/health"
    last_error: Exception | None = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Container did not become healthy: {last_error}")


async def verify(url: str) -> None:
    async with Client(url, timeout=10) as client:
        tools = await client.list_tools()
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


if __name__ == "__main__":
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"
    wait_until_healthy(endpoint)
    asyncio.run(verify(endpoint))
