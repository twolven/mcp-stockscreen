import json
from pathlib import Path

from stockscreen.domain import fundamental
from stockscreen.provider import normalize_news


def test_captured_equity_etf_index_hyphen_missing_news_calendar_financial_shapes():
    fixture = json.loads((Path(__file__).parent / "fixtures/yfinance_contracts.json").read_text())
    assert [fixture[key]["symbol"] for key in ("equity", "etf", "index", "hyphenated")] == [
        "AAPL",
        "SPY",
        "^GSPC",
        "BRK-B",
    ]
    assert not fundamental(fixture["missing_fundamentals"], {"max_pe": 20})[0]
    assert normalize_news(fixture["news"], 3650)[0]["publisher"] == "Wire"
    assert fixture["calendar"]["Earnings Date"] and fixture["financials"]
