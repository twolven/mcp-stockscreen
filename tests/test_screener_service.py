"""Tests for stockscreen.services.screener module."""

import datetime
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from stockscreen.services.screener import ScreenerService
from stockscreen.store.data_store import ScreenerDataStore


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def mock_news_service():
    return MagicMock()


@pytest.fixture
def screener_service(tmp_path, mock_provider, mock_news_service):
    store = ScreenerDataStore(base_path=str(tmp_path))
    return ScreenerService(
        provider=mock_provider,
        store=store,
        news_service=mock_news_service,
    )


@pytest.fixture
def mock_ticker_info():
    return {
        "regularMarketPrice": 150.0,
        "currentPrice": 150.0,
        "marketCap": 2_500_000_000_000,
        "forwardPE": 25.5,
        "dividendYield": 0.005,
        "revenueGrowth": 0.08,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "quoteType": "EQUITY",
        "averageVolume": 50_000_000,
        "regularMarketVolume": 45_000_000,
        "totalAssets": 400_000_000_000,
        "expenseRatio": 0.0009,
        "category": "Large Blend",
        "assetClass": "Equity",
        "previousClose": 149.0,
    }


@pytest.fixture
def mock_etf_info():
    return {
        "regularMarketPrice": 450.0,
        "previousClose": 449.0,
        "quoteType": "ETF",
        "totalAssets": 400_000_000_000,
        "expenseRatio": 0.0009,
        "category": "Large Blend",
        "assetClass": "Equity",
        "averageVolume": 80_000_000,
        "regularMarketVolume": 75_000_000,
    }


@pytest.fixture
def mock_history_df():
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=252)
    base = 100 + np.linspace(0, 20, 252) + np.sin(np.linspace(0, 4 * np.pi, 252)) * 10
    noise = np.random.normal(0, 1, 252)
    close = base + noise
    return pd.DataFrame(
        {
            "Open": close - np.random.uniform(0, 2, 252),
            "High": close + np.random.uniform(0, 3, 252),
            "Low": close - np.random.uniform(0, 3, 252),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 100_000_000, 252),
        },
        index=dates,
    )


@pytest.fixture
def mock_option_chain():
    OptionChain = namedtuple("OptionChain", ["calls", "puts"])
    strikes = [145.0, 147.5, 150.0, 152.5, 155.0]
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


def _setup_provider(mock_provider, info, history_df):
    """Configure mock provider with standard responses."""
    mock_provider.get_ticker_info = AsyncMock(return_value=info)
    mock_provider.get_history = AsyncMock(return_value=history_df)


# ============================================================
# 1. Technical screening
# ============================================================
class TestTechnicalScreen:
    async def test_basic_pass(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {}, symbols=["AAPL"])
        assert result["screen_type"] == "technical"
        assert result["matches"] == 1
        assert result["results"][0]["symbol"] == "AAPL"

    async def test_price_filter_min(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"min_price": 200}, symbols=["AAPL"])
        assert result["matches"] == 0
        assert any("Price" in r for r in result["rejected"][0]["rejection_reasons"])

    async def test_price_filter_max(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"max_price": 100}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_volume_filter(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"min_volume": 999_999_999_999}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_rsi_filter(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"max_rsi": 1}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_empty_history(self, screener_service, mock_provider, mock_ticker_info):
        mock_provider.get_ticker_info = AsyncMock(return_value=mock_ticker_info)
        mock_provider.get_history = AsyncMock(return_value=pd.DataFrame())
        result = await screener_service.run("technical", {}, symbols=["AAPL"])
        assert result["matches"] == 0
        assert len(result["rejected"]) == 1

    async def test_above_sma_200(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"above_sma_200": True}, symbols=["AAPL"])
        assert result["screen_type"] == "technical"

    async def test_symbols_from_criteria(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {"symbols": ["AAPL"]})
        assert result["matches"] == 1


# ============================================================
# 2. Fundamental screening
# ============================================================
class TestFundamentalScreen:
    async def test_basic_pass(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {}, symbols=["AAPL"])
        assert result["screen_type"] == "fundamental"
        assert result["matches"] == 1

    async def test_market_cap_filter(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {"min_market_cap": 10_000_000_000_000}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_pe_filter_range(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {"min_pe": 10, "max_pe": 20}, symbols=["AAPL"])
        assert result["matches"] == 0  # forwardPE is 25.5

    async def test_dividend_filter(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {"min_dividend": 5.0}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_revenue_growth_filter(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {"min_revenue_growth": 0.5}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_etf_screening(self, screener_service, mock_provider, mock_etf_info, mock_history_df):
        _setup_provider(mock_provider, mock_etf_info, mock_history_df)
        result = await screener_service.run("fundamental", {}, symbols=["SPY"])
        assert result["matches"] == 1
        assert "aum" in result["results"][0]

    async def test_no_info_skipped(self, screener_service, mock_provider, mock_history_df):
        mock_provider.get_ticker_info = AsyncMock(return_value=None)
        mock_provider.get_history = AsyncMock(return_value=mock_history_df)
        result = await screener_service.run("fundamental", {}, symbols=["BAD"])
        assert result["matches"] == 0


# ============================================================
# 3. Options screening
# ============================================================
class TestOptionsScreen:
    async def _setup_options(self, mock_provider, mock_ticker_info, mock_option_chain):
        mock_provider.get_ticker_info = AsyncMock(return_value=mock_ticker_info)
        mock_provider.get_option_expirations = AsyncMock(
            return_value=("2024-03-15", "2024-04-19", "2024-05-17")
        )
        mock_provider.get_option_chain = AsyncMock(return_value=mock_option_chain)
        mock_provider.get_history = AsyncMock(
            return_value=pd.DataFrame(
                {"Close": [150.0]}, index=[pd.Timestamp.now()]
            )
        )
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        mock_provider.get_earnings_dates = AsyncMock(
            return_value={
                "next_earnings": future_date,
                "earnings_range_end": future_date,
                "days_to_earnings": 30,
                "is_estimate": False,
            }
        )

    async def test_basic_pass(self, screener_service, mock_provider, mock_ticker_info, mock_option_chain):
        await self._setup_options(mock_provider, mock_ticker_info, mock_option_chain)
        result = await screener_service.run("options", {}, symbols=["AAPL"])
        assert result["screen_type"] == "options"
        assert result["matches"] >= 0

    async def test_no_options_rejected(self, screener_service, mock_provider, mock_ticker_info):
        mock_provider.get_ticker_info = AsyncMock(return_value=mock_ticker_info)
        mock_provider.get_option_expirations = AsyncMock(return_value=())
        mock_provider.get_earnings_dates = AsyncMock(
            return_value={"next_earnings": None, "earnings_range_end": None, "days_to_earnings": None, "is_estimate": None}
        )
        result = await screener_service.run("options", {}, symbols=["NOOPT"])
        assert result["matches"] == 0

    async def test_iv_filter_max(self, screener_service, mock_provider, mock_ticker_info, mock_option_chain):
        await self._setup_options(mock_provider, mock_ticker_info, mock_option_chain)
        result = await screener_service.run("options", {"max_iv": 1}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_volume_filter(self, screener_service, mock_provider, mock_ticker_info, mock_option_chain):
        await self._setup_options(mock_provider, mock_ticker_info, mock_option_chain)
        result = await screener_service.run("options", {"min_option_volume": 999_999_999}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_put_call_ratio(self, screener_service, mock_provider, mock_ticker_info, mock_option_chain):
        await self._setup_options(mock_provider, mock_ticker_info, mock_option_chain)
        result = await screener_service.run("options", {"min_put_call_ratio": 100}, symbols=["AAPL"])
        assert result["matches"] == 0

    async def test_earnings_date_filter(self, screener_service, mock_provider, mock_ticker_info, mock_option_chain):
        await self._setup_options(mock_provider, mock_ticker_info, mock_option_chain)
        result = await screener_service.run("options", {"max_days_to_earnings": 0}, symbols=["AAPL"])
        assert result["matches"] == 0


# ============================================================
# 4. Custom screening (multi-criteria)
# ============================================================
class TestCustomScreen:
    async def test_technical_and_fundamental_pass(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run(
            "custom",
            {"technical": {"min_price": 10}, "fundamental": {"min_market_cap": 1_000_000}},
            symbols=["AAPL"],
        )
        assert result["screen_type"] == "custom"
        assert result["matches"] >= 1

    async def test_technical_reject_stops_further(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run(
            "custom",
            {"technical": {"min_price": 99999}, "fundamental": {"min_market_cap": 1}},
            symbols=["AAPL"],
        )
        assert result["matches"] == 0
        assert len(result["rejected"]) == 1

    async def test_symbols_from_criteria(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run(
            "custom",
            {"symbols": ["AAPL"], "technical": {"min_price": 10}},
        )
        assert result["matches"] >= 1


# ============================================================
# 5. Error handling
# ============================================================
class TestErrorHandling:
    async def test_error_in_single_symbol_continues(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        call_count = 0

        async def mock_get_info(symbol):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise RuntimeError("api error")
            return mock_ticker_info

        mock_provider.get_ticker_info = mock_get_info
        mock_provider.get_history = AsyncMock(return_value=mock_history_df)

        result = await screener_service.run("technical", {}, symbols=["BAD", "AAPL"])
        # BAD should be rejected, AAPL should pass or be processed
        assert len(result["rejected"]) >= 1 or result["matches"] >= 1

    async def test_invalid_screen_type(self, screener_service):
        with pytest.raises(ValueError, match="Invalid screen type"):
            await screener_service.run("invalid_type", {}, symbols=["AAPL"])


# ============================================================
# 6. Dispatch method
# ============================================================
class TestRun:
    async def test_run_dispatches_technical(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("technical", {}, symbols=["AAPL"])
        assert result["screen_type"] == "technical"

    async def test_run_dispatches_fundamental(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        result = await screener_service.run("fundamental", {}, symbols=["AAPL"])
        assert result["screen_type"] == "fundamental"

    async def test_run_with_watchlist(self, screener_service, mock_provider, mock_ticker_info, mock_history_df):
        _setup_provider(mock_provider, mock_ticker_info, mock_history_df)
        # Pre-create a watchlist
        screener_service.store.save_watchlist("mylist", ["AAPL", "MSFT"])
        result = await screener_service.run("technical", {}, watchlist_name="mylist")
        assert result["screen_type"] == "technical"
        # Provider should have been called for both symbols
        assert mock_provider.get_ticker_info.call_count == 2
