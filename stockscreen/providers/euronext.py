"""Euronext provider — bidirectional ISIN ↔ ticker resolution with local cache."""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import requests

logger = logging.getLogger("stockscreen-server-v1")

# ---------------------------------------------------------------------------
# MIC → Yahoo Finance exchange suffix
# ---------------------------------------------------------------------------

_MIC_TO_SUFFIX: dict[str, str] = {
    "XPAR": ".PA",
    "XETR": ".DE",
    "XLON": ".L",
    "XAMS": ".AS",
    "XMIL": ".MI",
    "XMAD": ".MC",
    "XBRU": ".BR",
    "XLIS": ".LS",
    "XHEL": ".HE",
    "XSTO": ".ST",
    "XOSL": ".OL",
}

_EXCHANGE_SUFFIXES = {".PA", ".DE", ".L", ".AS", ".MI", ".MC", ".BR", ".LS", ".HE", ".ST", ".OL"}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://live.euronext.com/",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalize_ticker(ticker: str) -> str:
    """Strip exchange suffix and uppercase. 'TTE.PA' → 'TTE'."""
    ticker = ticker.upper()
    for suffix in _EXCHANGE_SUFFIXES:
        if ticker.endswith(suffix.upper()):
            return ticker[: -len(suffix)]
    return ticker


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EuronextRecord:
    isin: str
    symbol: str        # ex: "TTE"
    name: str          # ex: "TotalEnergies SE"
    mic: str           # ex: "XPAR"
    yahoo_ticker: str  # ex: "TTE.PA"
    cached_at: str     # ISO timestamp


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class EuronextProvider:
    """Bidirectional ISIN ↔ ticker resolution via the Euronext Live API.

    Args:
        cache_dir: Directory where ``euronext_{key}.json`` files are stored.
        cache_ttl_seconds: Seconds before a cache entry is considered stale.
            Defaults to 7 days (ticker/ISIN mappings rarely change).
        timeout: Per-request HTTP timeout in seconds.
    """

    _QUOTE_URL = (
        "https://live.euronext.com/api/quotes/{isin}/intraday_ioapi/2"
        "?fieldlist=isin,symbol,name,mic"
    )
    _SEARCH_URL = "https://live.euronext.com/search_instruments/{symbol}"

    def __init__(
        self,
        cache_dir: str,
        cache_ttl_seconds: float = 7 * 86400.0,
        timeout: float = 10.0,
    ):
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve_ticker(self, isin: str) -> EuronextRecord | None:
        """Resolve an ISIN to a Yahoo Finance ticker.

        Returns ``None`` if the ISIN is not found on Euronext.
        Falls back to stale cache on network error.
        """
        cache = self._load_cache(isin)
        if cache is not None and not self._is_expired(cache):
            return EuronextRecord(**cache["data"])

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: self._http_get(self._QUOTE_URL.format(isin=isin))
            )
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(f"[{isin}] EuronextProvider.resolve_ticker error: {exc}")
            if cache is not None:
                logger.warning(f"[{isin}] Serving stale cache.")
                return EuronextRecord(**cache["data"])
            return None

        if not data.get("isin"):
            return None

        rec = self._build_record(data)
        # Save under ISIN key
        self._save_cache(isin, rec)
        return rec

    async def resolve_isin(self, ticker: str) -> EuronextRecord | None:
        """Resolve a ticker (with or without exchange suffix) to an ISIN.

        Returns ``None`` if the ticker is not found on Euronext.
        Falls back to stale cache on network error.
        """
        normalized = _normalize_ticker(ticker)

        cache = self._load_cache(normalized)
        if cache is not None and not self._is_expired(cache):
            return EuronextRecord(**cache["data"])

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: self._http_get(self._SEARCH_URL.format(symbol=normalized))
            )
            results = json.loads(raw)
        except Exception as exc:
            logger.warning(f"[{normalized}] EuronextProvider.resolve_isin error: {exc}")
            if cache is not None:
                logger.warning(f"[{normalized}] Serving stale cache.")
                return EuronextRecord(**cache["data"])
            return None

        if not results or not isinstance(results, list):
            return None

        rec = self._build_record(results[0])
        # Save under both ticker key and ISIN key (shared cache)
        self._save_cache(normalized, rec)
        self._save_cache(rec.isin, rec)
        return rec

    def invalidate_cache(self, key: str) -> None:
        """Remove the cache file for the given ISIN or ticker (normalised)."""
        normalized = _normalize_ticker(key)
        for k in (key, normalized):
            path = self._cache_path(k)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"[{k}] Cache removal error: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _http_get(self, url: str) -> str:
        """Synchronous HTTP GET — always called via run_in_executor."""
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.text

    def _build_record(self, data: dict) -> EuronextRecord:
        mic = data.get("mic", "")
        symbol = data.get("symbol", "")
        suffix = _MIC_TO_SUFFIX.get(mic, "")
        return EuronextRecord(
            isin=data.get("isin", ""),
            symbol=symbol,
            name=data.get("name", ""),
            mic=mic,
            yahoo_ticker=f"{symbol}{suffix}",
            cached_at=_iso_now(),
        )

    def _cache_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"euronext_{key}.json")

    def _load_cache(self, key: str) -> dict | None:
        try:
            with open(self._cache_path(key)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"[{key}] Cache read error: {exc}")
            return None

    def _save_cache(self, key: str, rec: EuronextRecord) -> None:
        record = {"timestamp": time.time(), "data": asdict(rec)}
        try:
            with open(self._cache_path(key), "w") as f:
                json.dump(record, f, indent=2)
        except Exception as exc:
            logger.warning(f"[{key}] Cache write error: {exc}")

    def _is_expired(self, cache: dict) -> bool:
        return (time.time() - cache.get("timestamp", 0)) > self._cache_ttl


def _iso_now() -> str:
    import datetime
    return datetime.datetime.now().isoformat()
