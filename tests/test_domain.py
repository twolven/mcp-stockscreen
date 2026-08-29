from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from stockscreen.domain import (
    MISSING,
    days_until_earnings,
    fundamental,
    metric,
    news_matches,
    options_metrics,
    technical,
)
from stockscreen.models import validate_criteria


def history(length=220, fall=False):
    close = np.arange(1, length + 1, dtype=float)
    if fall:
        close = close[::-1]
    return pd.DataFrame(
        {"Close": close, "High": close + 1, "Low": close - 1, "Volume": [1_000_000] * length}
    )


def test_fundamental_all_legacy_bounds_and_missing():
    info = {
        "marketCap": 5e9,
        "trailingPE": 10,
        "dividendYield": 0.04,
        "revenueGrowth": 0.1,
        "totalAssets": 2e9,
        "annualReportExpenseRatio": 0.002,
        "regularMarketVolume": 2e6,
        "profitMargins": 0.2,
        "debtToEquity": 0.5,
        "priceToBook": 3.0,
    }
    criteria = {
        "min_market_cap": 1e9,
        "max_market_cap": 10e9,
        "min_pe": 5,
        "max_pe": 20,
        "min_dividend": 0.03,
        "min_revenue_growth": 0.05,
        "min_profit_margin": 0.1,
        "max_debt_to_equity": 1.0,
        "max_price_to_book": 5.0,
        "min_aum": 1e9,
        "max_expense_ratio": 0.01,
        "min_volume": 1e6,
    }
    ok, reasons, values = fundamental(info, criteria)
    assert ok and not reasons and values["pe"] == 10
    for key, value in criteria.items():
        ok, reasons, _ = fundamental({}, {key: value})
        assert not ok and reasons[0].startswith("missing metric")
    assert metric({}, "x") is MISSING and metric({"x": "bad"}, "x") is MISSING


def test_fundamental_failures_are_not_silent():
    ok, reasons, _ = fundamental(
        {"trailingPE": 500, "revenueGrowth": -0.9}, {"max_pe": 20, "min_revenue_growth": 0}
    )
    assert not ok and len(reasons) == 2
    assert fundamental({"trailingPE": 20}, {"max_pe": 20})[0]
    assert not fundamental({"trailingPE": 20.0001}, {"max_pe": 20})[0]
    assert fundamental({"trailingPE": 5}, {"min_pe": 5})[0]
    assert not fundamental({"trailingPE": 4.9999}, {"min_pe": 5})[0]
    assert not fundamental({"debtToEquity": 900}, {"max_debt_to_equity": 1.0})[0]


def test_technical_all_legacy_criteria():
    ok, reasons, data = technical(
        history(),
        {
            "min_price": 100,
            "max_price": 300,
            "min_volume": 1e5,
            "above_sma_50": True,
            "above_sma_200": True,
            "min_rsi": 70,
            "max_rsi": 100,
            "max_atr_pct": 5,
        },
    )
    assert ok and not reasons and data["sma_200"]
    ok, reasons, _ = technical(
        history(fall=True),
        {"above_sma_50": True, "above_sma_200": True, "min_rsi": 70, "min_volume": 2e6},
    )
    assert not ok and len(reasons) >= 4
    assert not technical(pd.DataFrame(), {})[0]
    assert not technical(pd.DataFrame({"Close": range(20)}), {"max_atr_pct": 5})[0]


def option_frames():
    calls = pd.DataFrame(
        {"impliedVolatility": [0.4, 0.5], "volume": [100, 200], "bid": [1, 2], "ask": [1.05, 2.05]}
    )
    puts = pd.DataFrame({"impliedVolatility": [0.6], "volume": [150], "bid": [1], "ask": [1.02]})
    return calls, puts


def test_options_all_legacy_criteria():
    calls, puts = option_frames()
    ok, reasons, values = options_metrics(
        calls,
        puts,
        {
            "min_iv": 30,
            "max_iv": 70,
            "min_option_volume": 400,
            "min_put_call_ratio": 0.4,
            "max_spread": 10,
            "min_days_to_earnings": 5,
            "max_days_to_earnings": 20,
            "min_days": 10,
            "max_days": 20,
        },
        10,
        15,
    )
    assert ok and not reasons and values["option_volume"] == 450
    ok, reasons, _ = options_metrics(
        calls,
        puts,
        {"min_iv": 90, "max_spread": 0.01, "max_days_to_earnings": 2, "min_days": 20},
        10,
        15,
    )
    assert not ok and len(reasons) == 4


def test_news_keywords_exclusions_age_and_management():
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    new = datetime.now(UTC).isoformat()
    articles = [
        {"title": "CEO appointed for growth", "summary": "alpha", "published_at": old},
        {"title": "lawsuit", "summary": "beta", "published_at": new},
    ]
    ok, _, details = news_matches(
        articles,
        {
            "keywords": ["ceo", "growth"],
            "require_all_keywords": True,
            "exclude_keywords": ["lawsuit"],
            "management_changes": True,
            "min_days": 2,
        },
    )
    assert ok and len(details["articles"]) == 1
    assert not news_matches(articles, {"keywords": ["missing"]})[0]


def test_calendar_and_criteria_validation():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert days_until_earnings({"Earnings Date": [pd.Timestamp("2026-01-11")]}, now) == 10
    assert days_until_earnings({}, now) is None
    validate_criteria("technical", {"above_sma_50": True})
    validate_criteria("custom", {"technical": {"min_rsi": 20}})
    with pytest.raises(ValueError):
        validate_criteria("fundamental", {"max_pe_ratio": 20})
    with pytest.raises(ValueError):
        validate_criteria("technical", {"category": "invalid"})
    with pytest.raises(TypeError):
        validate_criteria("custom", {"news": []})
