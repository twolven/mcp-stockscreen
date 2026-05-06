"""Integration tests — mock yfinance at the provider level, exercise the full stack."""

import datetime
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from stockscreen.providers.yahoo import YahooProvider
from stockscreen.services.news import NewsService
from stockscreen.services.screener import ScreenerService
from stockscreen.services.watchlist import WatchlistService
from stockscreen.store.data_store import ScreenerDataStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ScreenerDataStore(base_path=str(tmp_path))


@pytest.fixture
def provider():
    return MagicMock(spec=YahooProvider)


@pytest.fixture
def history_df():
    np.random.seed(0)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=252)
    close = 100 + np.linspace(0, 30, 252) + np.random.normal(0, 1, 252)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, 252),
        },
        index=dates,
    )


@pytest.fixture
def ticker_info():
    return {
        "regularMarketPrice": 130.0,
        "currentPrice": 130.0,
        "marketCap": 2_000_000_000_000,
        "forwardPE": 22.0,
        "dividendYield": 0.006,
        "revenueGrowth": 0.10,
        "sector": "Technology",
        "industry": "Software",
        "quoteType": "EQUITY",
        "averageVolume": 30_000_000,
        "regularMarketVolume": 28_000_000,
    }


@pytest.fixture
def option_chain():
    OptionChain = namedtuple("OptionChain", ["calls", "puts"])
    strikes = [125.0, 127.5, 130.0, 132.5, 135.0]
    calls = pd.DataFrame(
        {
            "strike": strikes,
            "bid": [6.0, 4.0, 2.5, 1.2, 0.5],
            "ask": [6.5, 4.5, 3.0, 1.5, 0.8],
            "volume": [1000, 2000, 5000, 3000, 1500],
            "openInterest": [5000, 8000, 15000, 10000, 6000],
            "impliedVolatility": [0.30, 0.28, 0.25, 0.27, 0.29],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": strikes,
            "bid": [0.3, 0.8, 1.8, 3.5, 5.5],
            "ask": [0.5, 1.0, 2.2, 4.0, 6.0],
            "volume": [800, 1500, 4000, 2500, 1200],
            "openInterest": [3000, 6000, 12000, 8000, 5000],
            "impliedVolatility": [0.31, 0.29, 0.26, 0.28, 0.30],
        }
    )
    return OptionChain(calls=calls, puts=puts)


# ---------------------------------------------------------------------------
# 1. Full technical screen
# ---------------------------------------------------------------------------

class TestTechnicalIntegration:
    async def test_complete_technical_screen(self, store, provider, history_df, ticker_info):
        """End-to-end: symbols → provider → screener → result stored."""
        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        result = await screener.run("technical", {"min_price": 50}, symbols=["AAPL", "MSFT"])

        assert result["screen_type"] == "technical"
        assert result["matches"] == 2
        assert all("symbol" in r for r in result["results"])
        assert all("rsi" in r for r in result["results"])

    async def test_result_persisted_and_reloaded(self, store, provider, history_df, ticker_info):
        """A stored screening result can be loaded back intact."""
        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        result = await screener.run("technical", {}, symbols=["AAPL"])
        store.save_screening_result("run1", result)

        loaded = store.load_screening_result("run1")
        assert loaded is not None
        assert loaded["matches"] == result["matches"]


# ---------------------------------------------------------------------------
# 2. Watchlist create → screen cycle
# ---------------------------------------------------------------------------

class TestWatchlistScreenCycle:
    async def test_create_watchlist_then_screen(self, store, provider, history_df, ticker_info):
        """Create a watchlist, then run a screen using that watchlist."""
        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        wl_service = WatchlistService(store=store)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        # Create watchlist
        await wl_service.create("tech", ["AAPL", "MSFT", "GOOGL"])

        # Screen using watchlist
        result = await screener.run("technical", {}, watchlist_name="tech")

        assert result["matches"] == 3
        assert provider.get_ticker_info.call_count == 3

    async def test_update_watchlist_affects_screen(self, store, provider, history_df, ticker_info):
        """After updating a watchlist, the screen uses the new symbols."""
        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        wl_service = WatchlistService(store=store)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        await wl_service.create("mylist", ["AAPL"])
        await wl_service.update("mylist", ["AAPL", "MSFT"])

        result = await screener.run("fundamental", {}, watchlist_name="mylist")
        assert result["matches"] == 2


# ---------------------------------------------------------------------------
# 3. Custom multi-criteria screen
# ---------------------------------------------------------------------------

class TestCustomScreenIntegration:
    async def test_technical_and_fundamental_combined(self, store, provider, history_df, ticker_info):
        """Custom screen applies technical then fundamental filters."""
        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        result = await screener.run(
            "custom",
            {
                "technical": {"min_price": 10, "above_sma_200": True},
                "fundamental": {"min_market_cap": 1_000_000_000, "max_pe": 50},
            },
            symbols=["AAPL"],
        )

        assert result["screen_type"] == "custom"
        assert result["matches"] == 1

    async def test_early_rejection_skips_further_screens(self, store, provider, history_df, ticker_info):
        """When technical screen rejects, fundamental is never called."""
        call_count = {"fundamental": 0}
        original_fundamental = ScreenerService._screen_single_fundamental

        async def patched_fundamental(self, symbol, criteria):
            call_count["fundamental"] += 1
            return await original_fundamental(self, symbol, criteria)

        provider.get_ticker_info = AsyncMock(return_value=ticker_info)
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)
        screener._screen_single_fundamental = lambda s, c: patched_fundamental(screener, s, c)

        result = await screener.run(
            "custom",
            {
                "technical": {"min_price": 999999},  # will reject
                "fundamental": {"min_market_cap": 1},
            },
            symbols=["AAPL"],
        )

        assert result["matches"] == 0
        assert call_count["fundamental"] == 0


# ---------------------------------------------------------------------------
# 4. Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    async def test_one_bad_symbol_does_not_stop_others(self, store, provider, history_df, ticker_info):
        """If one symbol raises, the rest are still processed."""
        call_count = [0]

        async def selective_info(symbol):
            call_count[0] += 1
            if symbol == "BAD":
                raise RuntimeError("network error")
            return ticker_info

        provider.get_ticker_info = selective_info
        provider.get_history = AsyncMock(return_value=history_df)

        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        result = await screener.run("technical", {}, symbols=["BAD", "AAPL", "MSFT"])

        assert result["matches"] == 2
        assert len(result["rejected"]) >= 1
        bad_rejected = [r for r in result["rejected"] if r["symbol"] == "BAD"]
        assert len(bad_rejected) == 1

    async def test_empty_symbol_list_returns_empty_result(self, store, provider):
        """An empty symbol list returns a valid result with 0 matches."""
        news = MagicMock(spec=NewsService)
        screener = ScreenerService(provider=provider, store=store, news_service=news)

        result = await screener.run("technical", {}, symbols=[])

        assert result["matches"] == 0
        assert result["results"] == []
        assert result["rejected"] == []
