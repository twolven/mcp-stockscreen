import pytest
from fastmcp import Client

from stockscreen.server import mcp


@pytest.mark.asyncio
async def test_exact_public_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == {
        "run_stock_screen",
        "get_stock_news",
        "manage_watchlist",
        "get_screening_result",
    }
