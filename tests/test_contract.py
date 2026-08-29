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
    screen = next(t for t in tools if t.name == "run_stock_screen")
    assert screen.inputSchema["properties"]["screen_type"]["enum"] == [
        "technical",
        "fundamental",
        "options",
        "news",
        "custom",
    ]
    action = next(t for t in tools if t.name == "manage_watchlist")
    assert action.inputSchema["properties"]["action"]["enum"] == [
        "create",
        "update",
        "delete",
        "get",
    ]
    assert {"success", "timestamp", "data", "provider", "warnings"} <= set(
        screen.outputSchema["properties"]
    )
