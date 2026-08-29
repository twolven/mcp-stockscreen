from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from stockscreen.provider import ProviderError, YahooProvider, clean, envelope, normalize_news


def test_retry_cache_failure(monkeypatch):
    p = YahooProvider(retries=1)
    calls = []

    def op():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError()
        return 2

    monkeypatch.setattr("stockscreen.provider.time.sleep", lambda _: None)
    assert p._call(("x",), op) == 2
    assert p._call(("x",), lambda: 3) == 2
    with pytest.raises(ProviderError):
        p._call(("y",), lambda: (_ for _ in ()).throw(OSError()))


def test_accessors_and_universe_payload(monkeypatch):
    class T:
        def __init__(self):
            self.options = ["2027-01-01"]
            self.news = []
            self.calendar = {}

        def get_info(self):
            return {"x": 1}

        def history(self, **kwargs):
            return kwargs

        def option_chain(self, x):
            return x

    captured = []
    monkeypatch.setattr(
        "stockscreen.provider.yf.screen",
        lambda query, **kwargs: captured.append(kwargs) or {"quotes": [{"symbol": "AAPL"}]},
    )
    p = YahooProvider(retries=0)
    t = T()
    assert p.info("X", t)
    assert p.history("X", t)["timeout"] == 15
    assert p.expirations("X", t)
    assert p.chain("X", "2027", t) == "2027"
    assert p.news("X", t) == []
    assert p.calendar("X", t) == {}
    assert (
        p.universe(300) == ["AAPL"]
        and captured[-1]["size"] == 250
        and captured[-1]["sortAsc"] is False
    )
    assert p.universe(250, "etf") == ["AAPL"] and captured[-1]["sortField"] == "fundnetassets"


def test_normalize_news_and_clean():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    items = [
        {
            "content": {
                "title": "x",
                "pubDate": "2026-01-01T00:00:00Z",
                "provider": {"displayName": "Wire"},
                "canonicalUrl": {"url": "https://x"},
            }
        },
        {"providerPublishTime": (now - timedelta(days=10)).timestamp()},
    ]
    news = normalize_news(items, 2, now)
    assert len(news) == 1 and news[0]["publisher"] == "Wire"
    value = clean({"stamp": pd.Timestamp("2026-01-01"), "bad": float("nan"), "number": np_int()})
    assert value["bad"] is None and value["number"] == 1
    assert envelope({})["success"]


def np_int():
    return pd.Series([1]).iloc[0]
