"""Shared fixtures for stockscreen tests."""

import os
import sys
import tempfile

# Set env var BEFORE importing stockscreen to prevent migration side effects
# and to redirect data to a temp directory
_test_data_dir = tempfile.mkdtemp(prefix="stockscreen_test_")
os.environ["STOCKSCREEN_DATA_PATH"] = _test_data_dir

import datetime
import json
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_data_path(tmp_path):
    """Provide a fresh temp directory for ScreenerDataStore tests."""
    return tmp_path


@pytest.fixture
def mock_ticker_info():
    """Return a dict mimicking yf.Ticker("AAPL").info for an equity."""
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
def mock_etf_info():
    """Return a dict mimicking yf.Ticker("SPY").info for an ETF."""
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
    """Return a synthetic 252-row OHLCV DataFrame for 1 year of trading."""
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=252)
    # Trending upward with oscillation
    base = 100 + np.linspace(0, 20, 252) + np.sin(np.linspace(0, 4 * np.pi, 252)) * 10
    noise = np.random.normal(0, 1, 252)
    close = base + noise

    df = pd.DataFrame(
        {
            "Open": close - np.random.uniform(0, 2, 252),
            "High": close + np.random.uniform(0, 3, 252),
            "Low": close - np.random.uniform(0, 3, 252),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 100_000_000, 252),
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_news_items():
    """Return a list of dicts matching yfinance news format."""
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
            "title": "SEC investigation into accounting practices",
            "publisher": "Bloomberg",
            "providerPublishTime": int((now - datetime.timedelta(days=5)).timestamp()),
            "type": "STORY",
            "summary": "Federal regulators are probing the company.",
        },
        {
            "title": "Quarterly earnings beat expectations",
            "publisher": "CNBC",
            "providerPublishTime": int((now - datetime.timedelta(days=10)).timestamp()),
            "type": "STORY",
            "summary": "Revenue and profit exceeded analyst estimates.",
        },
    ]


@pytest.fixture
def mock_option_chain():
    """Return a namedtuple-like object with .calls and .puts DataFrames."""
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
