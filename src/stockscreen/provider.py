import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from .models import ResponseEnvelope


class ProviderError(RuntimeError):
    pass


class YahooProvider:
    def __init__(self, retries: int = 2, cache_seconds: float = 30, timeout: float = 15):
        self.retries = retries
        self.cache_seconds = cache_seconds
        self.timeout = timeout
        self._cache: dict[tuple[str, ...], tuple[float, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stockscreen-yahoo")

    def _call(self, key: tuple[str, ...], operation: Callable[[], Any], cache: bool = True):
        hit = self._cache.get(key)
        if cache and hit and time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        last = None
        for attempt in range(self.retries + 1):
            try:
                value = self._executor.submit(operation).result(timeout=self.timeout)
                if cache:
                    self._cache[key] = (time.monotonic(), value)
                return value
            except (TimeoutError, ConnectionError, OSError, YFRateLimitError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.1 * 2**attempt)
            except YFException as exc:
                raise ProviderError(str(exc)) from exc
        assert isinstance(last, BaseException)
        raise ProviderError(str(last)) from last

    def ticker(self, symbol):
        return yf.Ticker(symbol)

    def info(self, symbol, ticker):
        return self._call((symbol, "info"), ticker.get_info)

    def history(self, symbol, ticker):
        return self._call(
            (symbol, "history"),
            lambda: ticker.history(period="1y", auto_adjust=False, repair=True, timeout=15),
        )

    def expirations(self, symbol, ticker):
        return list(self._call((symbol, "options"), lambda: ticker.options))

    def chain(self, symbol, expiration, ticker):
        return self._call((symbol, "chain", expiration), lambda: ticker.option_chain(expiration))

    def news(self, symbol, ticker):
        return self._call((symbol, "news"), lambda: ticker.news)

    def calendar(self, symbol, ticker):
        return self._call((symbol, "calendar"), lambda: ticker.calendar)

    def universe(self, size=250, category=None):
        if category == "etf":
            query = yf.ETFQuery(
                "and",
                [yf.ETFQuery("eq", ["region", "us"]), yf.ETFQuery("gt", ["fundnetassets", 0])],
            )
            sort_field = "fundnetassets"
        else:
            bounds = {
                "mega_cap": (200e9, None),
                "large_cap": (10e9, 200e9),
                "mid_cap": (2e9, 10e9),
                "small_cap": (300e6, 2e9),
                "micro_cap": (0, 300e6),
            }
            minimum, maximum = bounds.get(category, (0, None))
            operands = [
                yf.EquityQuery("eq", ["region", "us"]),
                yf.EquityQuery("gt", ["intradaymarketcap", minimum]),
            ]
            if maximum is not None:
                operands.append(yf.EquityQuery("lt", ["intradaymarketcap", maximum]))
            query = yf.EquityQuery("and", operands)
            sort_field = "intradaymarketcap"
        response = self._call(
            ("universe", str(size), str(category)),
            lambda: yf.screen(query, size=min(size, 250), sortField=sort_field, sortAsc=False),
        )
        return [quote["symbol"] for quote in response.get("quotes", []) if quote.get("symbol")]


def clean(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def envelope(data, warnings=None) -> ResponseEnvelope:
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
        source = content.get("provider") or {}
        thumbnail = content.get("thumbnail") or {}
        canonical = content.get("canonicalUrl") or {}
        out.append(
            {
                "title": content.get("title"),
                "publisher": source.get("displayName") if isinstance(source, dict) else source,
                "published_at": datetime.fromtimestamp(stamp, UTC).isoformat(),
                "summary": content.get("summary") or content.get("description"),
                "url": canonical.get("url") if isinstance(canonical, dict) else content.get("link"),
                "thumbnail": thumbnail.get("originalUrl") if isinstance(thumbnail, dict) else None,
            }
        )
    return out
