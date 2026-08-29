from datetime import UTC, datetime

import pandas as pd

from stockscreen.domain import MISSING, fundamental, metric, technical
from stockscreen.provider import normalize_news


def test_missing_metric_is_rejected():
    ok, reasons = fundamental({}, {"max_pe_ratio": 20})
    assert not ok and reasons == ["missing metric: pe_ratio"] and metric({}, "x") is MISSING


def test_fundamental_bounds():
    assert fundamental({"trailingPE": 10}, {"max_pe_ratio": 20}) == (True, [])
    assert not fundamental({"trailingPE": 30}, {"max_pe_ratio": 20})[0]


def test_technical():
    frame = pd.DataFrame({"Close": range(1, 61), "Volume": [100] * 60})
    ok, reasons, data = technical(frame, {"above_sma_20": True, "min_price": 50})
    assert ok and not reasons and data["sma_20"] == 50.5


def test_news_current_shape():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    items = [
        {
            "content": {
                "title": "x",
                "pubDate": "2026-01-01T00:00:00Z",
                "provider": {"displayName": "Wire"},
                "canonicalUrl": {"url": "https://x"},
            }
        }
    ]
    assert normalize_news(items, 2, now)[0]["publisher"] == "Wire"
