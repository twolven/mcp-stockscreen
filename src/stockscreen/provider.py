from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf


class YahooProvider:
    def ticker(self, symbol):
        return yf.Ticker(symbol)

    def universe(self, count=250):
        query = yf.EquityQuery("gt", ["intradaymarketcap", 0])
        response = yf.screen(query, count=min(count, 250))
        return [q["symbol"] for q in response.get("quotes", []) if q.get("symbol")]


def clean(v: Any) -> Any:
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if v is None or (not isinstance(v, (dict, list)) and pd.isna(v)):
        return None
    return v


def envelope(data, warnings=None):
    now = datetime.now(UTC).isoformat()
    return {
        "success": True,
        "timestamp": now,
        "data": clean(data),
        "provider": {"name": "Yahoo Finance via yfinance", "as_of": now, "real_time": False},
        "warnings": warnings or [],
    }


def normalize_news(items, days_back, now=None):
    now = now or datetime.now(UTC)
    cutoff = now.timestamp() - days_back * 86400
    out = []
    for item in items or []:
        content = item.get("content", item)
        published = content.get("pubDate") or content.get("providerPublishTime")
        try:
            stamp = (
                datetime.fromisoformat(str(published).replace("Z", "+00:00")).timestamp()
                if isinstance(published, str)
                else float(published)
            )
        except (ValueError, TypeError):
            continue
        if stamp < cutoff:
            continue
        provider = content.get("provider") or {}
        thumbnail = content.get("thumbnail") or {}
        out.append(
            {
                "title": content.get("title"),
                "publisher": provider.get("displayName")
                if isinstance(provider, dict)
                else provider,
                "published_at": datetime.fromtimestamp(stamp, UTC).isoformat(),
                "summary": content.get("summary") or content.get("description"),
                "url": content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else content.get("link"),
                "thumbnail": thumbnail.get("originalUrl") if isinstance(thumbnail, dict) else None,
            }
        )
    return out
