"""Tests for stockscreen.server — FastMCP wiring."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stockscreen.server import create_services, mcp


# ============================================================
# Helpers
# ============================================================

def _make_services(tmp_path):
    """Return (provider, store, screener, watchlist, news) mocks via create_services override."""
    from stockscreen.store.data_store import ScreenerDataStore
    from stockscreen.services.screener import ScreenerService
    from stockscreen.services.watchlist import WatchlistService
    from stockscreen.services.news import NewsService

    provider = MagicMock()
    store = ScreenerDataStore(base_path=str(tmp_path))
    news = MagicMock(spec=NewsService)
    screener = MagicMock(spec=ScreenerService)
    watchlist = MagicMock(spec=WatchlistService)
    return provider, store, screener, watchlist, news


# ============================================================
# 1. Server initialisation
# ============================================================

class TestServerInit:
    def test_fastmcp_instance_exists(self):
        """The module-level `mcp` object is a FastMCP instance."""
        from mcp.server.fastmcp import FastMCP
        assert isinstance(mcp, FastMCP)

    def test_create_services_returns_tuple(self, tmp_path):
        """create_services() returns a tuple of service objects."""
        with patch("stockscreen.server.DEFAULT_DATA_PATH", str(tmp_path)):
            result = create_services()
        assert isinstance(result, tuple)
        assert len(result) == 4  # screener, watchlist, news, symbol_svc


# ============================================================
# 2. Tool routing — run_stock_screen
# ============================================================

class TestRunStockScreen:
    async def test_routes_to_screener(self, tmp_path):
        """run_stock_screen tool calls screener_service.run."""
        from stockscreen.server import run_stock_screen

        mock_screener = AsyncMock()
        mock_screener.run = AsyncMock(return_value={"screen_type": "technical", "matches": 0, "results": [], "rejected": []})

        with patch("stockscreen.server._screener", mock_screener):
            result = await run_stock_screen(screen_type="technical", criteria={})

        mock_screener.run.assert_called_once()
        assert "screen_type" in result

    async def test_invalid_screen_type_returns_error(self, tmp_path):
        """An invalid screen_type returns an error dict instead of raising."""
        from stockscreen.server import run_stock_screen

        mock_screener = AsyncMock()
        mock_screener.run = AsyncMock(side_effect=ValueError("Invalid screen type: bad"))

        with patch("stockscreen.server._screener", mock_screener):
            result = await run_stock_screen(screen_type="bad", criteria={})

        assert "error" in result

    async def test_save_result_persists(self, tmp_path):
        """When save_result is provided, the result is stored via the screener's store."""
        from stockscreen.server import run_stock_screen
        from stockscreen.store.data_store import ScreenerDataStore

        store = ScreenerDataStore(base_path=str(tmp_path))
        screen_result = {
            "screen_type": "technical",
            "matches": 1,
            "results": [{"symbol": "AAPL"}],
            "rejected": [],
        }
        mock_screener = AsyncMock()
        mock_screener.run = AsyncMock(return_value=screen_result)
        mock_screener.store = store

        with patch("stockscreen.server._screener", mock_screener):
            await run_stock_screen(screen_type="technical", criteria={}, save_result="myresult")

        saved = store.load_screening_result("myresult")
        assert saved is not None
        assert saved["matches"] == 1


# ============================================================
# 3. Tool routing — get_stock_news
# ============================================================

class TestGetStockNews:
    async def test_routes_to_news_service(self):
        """get_stock_news calls news_service.get_news_data."""
        from stockscreen.server import get_stock_news

        mock_news = AsyncMock()
        mock_news.get_news_data = AsyncMock(return_value={"recent_news": [], "key_events": []})

        with patch("stockscreen.server._news", mock_news):
            result = await get_stock_news(symbol="AAPL")

        mock_news.get_news_data.assert_called_once_with("AAPL", days_back=30)
        assert "recent_news" in result


# ============================================================
# 4. Tool routing — manage_watchlist
# ============================================================

class TestManageWatchlist:
    async def test_create_action(self):
        """manage_watchlist create action calls watchlist_service.dispatch."""
        from stockscreen.server import manage_watchlist

        mock_wl = AsyncMock()
        mock_wl.dispatch = AsyncMock(return_value={"message": "created"})

        with patch("stockscreen.server._watchlist", mock_wl):
            result = await manage_watchlist(action="create", name="mylist", symbols=["AAPL"])

        mock_wl.dispatch.assert_called_once_with("create", "mylist", ["AAPL"])
        assert "message" in result

    async def test_validation_error_returns_error(self):
        """A ValidationError from watchlist_service is returned as error dict."""
        from stockscreen.server import manage_watchlist
        from stockscreen.exceptions import ValidationError

        mock_wl = AsyncMock()
        mock_wl.dispatch = AsyncMock(side_effect=ValidationError("bad name"))

        with patch("stockscreen.server._watchlist", mock_wl):
            result = await manage_watchlist(action="create", name="!!!", symbols=[])

        assert "error" in result


# ============================================================
# 5. Tool routing — get_screening_result
# ============================================================

class TestGetScreeningResult:
    async def test_returns_saved_result(self, tmp_path):
        """get_screening_result loads from the screener's store."""
        from stockscreen.server import get_screening_result
        from stockscreen.store.data_store import ScreenerDataStore

        store = ScreenerDataStore(base_path=str(tmp_path))
        store.save_screening_result("run1", {"matches": 5})

        mock_screener = MagicMock()
        mock_screener.store = store

        with patch("stockscreen.server._screener", mock_screener):
            result = await get_screening_result(name="run1")

        assert result["matches"] == 5

    async def test_missing_result_returns_error(self, tmp_path):
        """get_screening_result returns error dict when result not found."""
        from stockscreen.server import get_screening_result
        from stockscreen.store.data_store import ScreenerDataStore

        store = ScreenerDataStore(base_path=str(tmp_path))
        mock_screener = MagicMock()
        mock_screener.store = store

        with patch("stockscreen.server._screener", mock_screener):
            result = await get_screening_result(name="nonexistent")

        assert "error" in result
