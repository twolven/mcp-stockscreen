"""MarketDataFacade — unified access to Yahoo, Boursorama, and Euronext providers."""

import asyncio
import logging
from typing import Any

import pandas as pd

from stockscreen.exceptions import APIError
from stockscreen.providers.boursorama import BoursoramaProvider
from stockscreen.providers.euronext import EuronextProvider, _normalize_ticker
from stockscreen.providers.yahoo import YahooProvider

logger = logging.getLogger("stockscreen-server-v1")

# ISIN pattern: 2 uppercase letters + 10 alphanumerics
_ISIN_PREFIX_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _looks_like_isin(identifier: str) -> bool:
    """Heuristic: ISIN is 12 chars, starts with 2 letters."""
    return (
        len(identifier) == 12
        and identifier[:2].upper().isalpha()
        and identifier[2:].isalnum()
    )


def _dividend_yield_pct(info: dict) -> float:
    """Return dividend yield as a plain percentage from Yahoo info dict."""
    price = info.get("regularMarketPrice") or info.get("currentPrice") or 0
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0
    if rate and price:
        return rate / price * 100
    raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
    if raw > 1.0:
        return float(raw)
    return raw * 100


class MarketDataFacade:
    """Single entry point for all market data.

    Orchestrates three providers:
    - ``YahooProvider``     — prices, history, options, news, fundamentals
    - ``BoursoramaProvider``— dividend, yield, consensus, financials (Euronext)
    - ``EuronextProvider``  — ISIN ↔ ticker resolution

    Dividend strategy: **Boursorama-first**, Yahoo fallback.
    All other fields come from Yahoo.

    Args:
        yahoo:      ``YahooProvider`` instance.
        boursorama: ``BoursoramaProvider`` instance.
        euronext:   ``EuronextProvider`` instance.
    """

    def __init__(
        self,
        yahoo: YahooProvider,
        boursorama: BoursoramaProvider,
        euronext: EuronextProvider,
    ):
        self._yahoo = yahoo
        self._boursorama = boursorama
        self._euronext = euronext

    # ------------------------------------------------------------------
    # Public API — drop-in replacement for YahooProvider in services
    # ------------------------------------------------------------------

    async def get_ticker_info(self, identifier: str) -> dict | None:
        """Enriched ticker info dict — alias for get_quote used by ScreenerService."""
        return await self.get_quote(identifier)

    async def get_quote(self, identifier: str) -> dict:
        """Fetch and merge data from Yahoo and Boursorama.

        Args:
            identifier: Yahoo ticker (``"TTE.PA"``) or ISIN (``"FR0000131104"``).

        Returns:
            Flat dict with all fields from Yahoo enriched with Boursorama
            dividend/consensus/performance data.

        Raises:
            APIError: if the identifier is an ISIN that cannot be resolved.
        """
        yahoo_ticker, bourso_query = await self._resolve(identifier)

        # Launch Yahoo and Boursorama in parallel
        yahoo_task = self._yahoo.get_ticker_info(yahoo_ticker)
        bourso_task = self._fetch_boursorama(bourso_query)

        yahoo_info, bourso_quote = await asyncio.gather(yahoo_task, bourso_task)

        return self._merge(yahoo_info or {}, bourso_quote)

    async def get_history(self, identifier: str, period: str = "1y") -> pd.DataFrame:
        """Get historical OHLCV — delegates to Yahoo."""
        yahoo_ticker, _ = await self._resolve(identifier)
        return await self._yahoo.get_history(yahoo_ticker, period=period)

    async def get_news(self, identifier: str) -> list[dict]:
        """Get recent news — delegates to Yahoo."""
        yahoo_ticker, _ = await self._resolve(identifier)
        return await self._yahoo.get_news(yahoo_ticker)

    async def get_option_chain(self, identifier: str, expiry: str) -> Any:
        """Get options chain — delegates to Yahoo."""
        yahoo_ticker, _ = await self._resolve(identifier)
        return await self._yahoo.get_option_chain(yahoo_ticker, expiry)

    async def get_option_expirations(self, identifier: str) -> tuple:
        """Get option expiration dates — delegates to Yahoo."""
        yahoo_ticker, _ = await self._resolve(identifier)
        return await self._yahoo.get_option_expirations(yahoo_ticker)

    async def get_earnings_dates(self, identifier: str) -> dict:
        """Get earnings dates — delegates to Yahoo."""
        yahoo_ticker, _ = await self._resolve(identifier)
        return await self._yahoo.get_earnings_dates(yahoo_ticker)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve(self, identifier: str) -> tuple[str, str]:
        """Return (yahoo_ticker, bourso_query) for any identifier.

        - Yahoo ticker (``"TTE.PA"``) → yahoo_ticker = ``"TTE.PA"``,
          bourso_query = ``"TTE"`` (suffix stripped)
        - ISIN → resolved via EuronextProvider; bourso_query = ISIN itself
        """
        if _looks_like_isin(identifier):
            record = await self._euronext.resolve_ticker(identifier)
            if record is None:
                raise APIError(
                    f"[{identifier}] Cannot resolve ISIN to Yahoo ticker via Euronext."
                )
            return record.yahoo_ticker, identifier  # Boursorama gets the ISIN

        # Plain ticker or Yahoo-format ticker
        yahoo_ticker = identifier
        bourso_query = _normalize_ticker(identifier)  # strip .PA / .DE / ...
        return yahoo_ticker, bourso_query

    async def _fetch_boursorama(self, query: str):
        """Fetch Boursorama quote, returning None on any error."""
        try:
            return await self._boursorama.get_quote(query)
        except Exception as exc:
            logger.debug(f"[{query}] Boursorama fetch skipped: {exc}")
            return None

    def _merge(self, yahoo_info: dict, bourso_quote) -> dict:
        """Merge Yahoo info dict with Boursorama quote.

        Boursorama-first for: dividende, rendement, last_dividend_date,
        consensus, performance.
        Everything else comes from Yahoo.
        """
        result = dict(yahoo_info)

        # ---- Dividend fields — Boursorama-first ----
        if bourso_quote is not None:
            dividende = bourso_quote.dividende
            rendement = bourso_quote.rendement
        else:
            dividende = None
            rendement = None

        if dividende is not None and rendement is not None:
            result["dividende"] = dividende
            result["rendement"] = rendement
        else:
            # Fallback: compute from Yahoo data
            result["dividende"] = yahoo_info.get("dividendRate") or \
                                   yahoo_info.get("trailingAnnualDividendRate")
            result["rendement"] = _dividend_yield_pct(yahoo_info)

        result["last_dividend_date"] = (
            bourso_quote.last_dividend_date if bourso_quote is not None else None
        )

        # ---- Boursorama-only fields ----
        result["consensus"] = bourso_quote.consensus if bourso_quote is not None else None
        result["performance"] = bourso_quote.performance if bourso_quote is not None else []

        return result
