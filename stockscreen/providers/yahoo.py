"""Yahoo Finance provider — async wrapper around yfinance."""

import asyncio
import datetime
import logging
from functools import wraps
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger("stockscreen-server-v1")

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def _retry(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_error
        return wrapper
    return decorator


class YahooProvider:
    """Async wrapper around yfinance. All network I/O runs in an executor."""

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Create a yfinance Ticker (synchronous)."""
        return yf.Ticker(symbol)

    @_retry()
    async def get_ticker_info(self, symbol: str) -> dict | None:
        """Get ticker info dict."""
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None, lambda: self._get_ticker(symbol).info
        )
        return info if info else None

    @_retry()
    async def get_history(
        self, symbol: str, period: str = "1y"
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._get_ticker(symbol).history(period=period)
        )

    @_retry()
    async def get_option_chain(self, symbol: str, expiry: str) -> Any:
        """Get options chain for a specific expiration date."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._get_ticker(symbol).option_chain(expiry)
        )

    @_retry()
    async def get_option_expirations(self, symbol: str) -> tuple:
        """Get available option expiration dates."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._get_ticker(symbol).options
        )

    @_retry()
    async def get_news(self, symbol: str) -> list[dict]:
        """Get recent news for a symbol."""
        loop = asyncio.get_event_loop()
        news = await loop.run_in_executor(
            None, lambda: self._get_ticker(symbol).news
        )
        return news if news else []

    async def get_earnings_dates(self, symbol: str) -> dict:
        """Get earnings dates for a symbol.

        Returns dict with next_earnings, earnings_range_end,
        days_to_earnings, is_estimate.
        """
        try:
            loop = asyncio.get_event_loop()
            calendar = await loop.run_in_executor(
                None, lambda: self._get_ticker(symbol).calendar
            )

            next_dates = []
            if calendar is not None and isinstance(calendar, dict):
                earnings_date = calendar.get("Earnings Date")
                if isinstance(earnings_date, list):
                    next_dates = earnings_date

            days_to_earnings = None
            if next_dates:
                earliest_date = next_dates[0]
                days_to_earnings = (earliest_date - datetime.datetime.now().date()).days

            return {
                "next_earnings": next_dates[0] if next_dates else None,
                "earnings_range_end": (
                    next_dates[1]
                    if len(next_dates) > 1
                    else next_dates[0] if next_dates else None
                ),
                "days_to_earnings": days_to_earnings,
                "is_estimate": len(next_dates) > 1 if next_dates else None,
            }
        except Exception as e:
            logger.error(f"Error getting earnings dates for {symbol}: {e}")
            return {
                "next_earnings": None,
                "earnings_range_end": None,
                "days_to_earnings": None,
                "is_estimate": None,
            }
