"""Tests for stockscreen.providers.yahoo module."""

import asyncio
import datetime
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stockscreen.providers.yahoo import YahooProvider


@pytest.fixture
def provider():
    return YahooProvider()


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
        "longBusinessSummary": "Apple Inc. designs consumer electronics.",
        "website": "https://apple.com",
        "companyOfficers": [
            {"name": "Tim Cook", "title": "CEO", "yearStarted": 2011},
        ],
        "exchange": "NMS",
        "averageVolume": 50_000_000,
        "regularMarketVolume": 45_000_000,
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


@pytest.fixture
def mock_news_items():
    now = datetime.datetime.now()
    return [
        {
            "title": "CEO announces new strategy for growth",
            "publisher": "Reuters",
            "providerPublishTime": int((now - datetime.timedelta(days=2)).timestamp()),
            "type": "STORY",
            "summary": "The CEO outlined plans for expansion.",
        },
        {
            "title": "Quarterly earnings beat expectations",
            "publisher": "CNBC",
            "providerPublishTime": int((now - datetime.timedelta(days=10)).timestamp()),
            "type": "STORY",
            "summary": "Revenue and profit exceeded analyst estimates.",
        },
    ]


# ============================================================
# 1. get_ticker_info
# ============================================================
class TestGetTickerInfo:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_info_dict(self, mock_ticker_cls, provider, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_ticker_info("AAPL")
        assert result["regularMarketPrice"] == 150.0
        assert result["sector"] == "Technology"

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_none_on_empty_info(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.info = None
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_ticker_info("BAD")
        assert result is None


# ============================================================
# 2. get_history
# ============================================================
class TestGetHistory:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_dataframe(self, mock_ticker_cls, provider, mock_history_df):
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_history_df
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_history("AAPL", period="1y")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 252
        assert "Close" in result.columns

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_empty_df_on_error(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_history("BAD")
        assert result.empty


# ============================================================
# 3. get_option_chain
# ============================================================
class TestGetOptionChain:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_chain(self, mock_ticker_cls, provider, mock_option_chain):
        mock_instance = MagicMock()
        mock_instance.option_chain.return_value = mock_option_chain
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_option_chain("AAPL", "2024-03-15")
        assert hasattr(result, "calls")
        assert hasattr(result, "puts")
        assert len(result.calls) == 5


# ============================================================
# 4. get_option_expirations
# ============================================================
class TestGetOptionExpirations:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_expirations(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.options = ("2024-03-15", "2024-04-19", "2024-05-17")
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_option_expirations("AAPL")
        assert len(result) == 3

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_empty_when_no_options(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.options = ()
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_option_expirations("NOOPT")
        assert result == ()


# ============================================================
# 5. get_news
# ============================================================
class TestGetNews:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_news_list(self, mock_ticker_cls, provider, mock_news_items):
        mock_instance = MagicMock()
        mock_instance.news = mock_news_items
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_news("AAPL")
        assert len(result) == 2
        assert result[0]["title"] == "CEO announces new strategy for growth"

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_returns_empty_list_when_no_news(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.news = []
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_news("AAPL")
        assert result == []


# ============================================================
# 6. get_earnings_dates
# ============================================================
class TestGetEarningsDates:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_with_earnings_dates(self, mock_ticker_cls, provider):
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        mock_instance = MagicMock()
        mock_instance.calendar = {"Earnings Date": [future_date]}
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_earnings_dates("AAPL")
        assert result["next_earnings"] == future_date
        assert result["days_to_earnings"] > 0

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_no_calendar(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.calendar = None
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_earnings_dates("AAPL")
        assert result["next_earnings"] is None
        assert result["days_to_earnings"] is None

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_earnings_date_range(self, mock_ticker_cls, provider):
        date1 = datetime.date.today() + datetime.timedelta(days=25)
        date2 = datetime.date.today() + datetime.timedelta(days=30)
        mock_instance = MagicMock()
        mock_instance.calendar = {"Earnings Date": [date1, date2]}
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_earnings_dates("AAPL")
        assert result["next_earnings"] == date1
        assert result["earnings_range_end"] == date2
        assert result["is_estimate"] is True

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_exception_handling(self, mock_ticker_cls, provider):
        mock_instance = MagicMock()
        mock_instance.calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("API error")))
        type(mock_instance).calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("API error")))
        mock_ticker_cls.return_value = mock_instance

        result = await provider.get_earnings_dates("AAPL")
        assert result["next_earnings"] is None


# ============================================================
# 7. run_in_executor usage
# ============================================================
class TestAsyncExecution:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_get_ticker_info_uses_executor(self, mock_ticker_cls, provider, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance

        with patch.object(asyncio.get_event_loop(), "run_in_executor", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_ticker_info
            result = await provider.get_ticker_info("AAPL")
            mock_exec.assert_called_once()


# ============================================================
# 8. retry behavior
# ============================================================
class TestRetryBehavior:
    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_retries_on_failure(self, mock_ticker_cls, provider):
        call_count = 0

        def side_effect(symbol):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            mock_inst = MagicMock()
            mock_inst.info = {"regularMarketPrice": 150.0}
            return mock_inst

        mock_ticker_cls.side_effect = side_effect

        with patch("stockscreen.providers.yahoo.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.get_ticker_info("AAPL")
            assert result["regularMarketPrice"] == 150.0
            assert call_count == 3

    @patch("stockscreen.providers.yahoo.yf.Ticker")
    async def test_exhausted_retries_raises(self, mock_ticker_cls, provider):
        mock_ticker_cls.side_effect = ConnectionError("permanent failure")

        with patch("stockscreen.providers.yahoo.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                await provider.get_ticker_info("AAPL")
