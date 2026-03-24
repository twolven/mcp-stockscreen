"""FastMCP server — wires tools to services."""

import asyncio
import logging

from mcp.server.fastmcp import FastMCP

from stockscreen.config import (
    DEFAULT_DATA_PATH,
    REFRESH_ON_STARTUP,
    SYMBOL_REFRESH_INTERVAL_HOURS,
    SYMBOL_SOURCES,
    setup_logging,
)
from stockscreen.exceptions import ValidationError
from stockscreen.providers.symbol_fetchers.registry import build_fetchers
from stockscreen.providers.yahoo import YahooProvider
from stockscreen.services.news import NewsService
from stockscreen.services.screener import ScreenerService
from stockscreen.services.symbol_service import SymbolService
from stockscreen.services.watchlist import WatchlistService
from stockscreen.store.data_store import ScreenerDataStore

logger = logging.getLogger("stockscreen-server-v1")

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP("stockscreen")

# ---------------------------------------------------------------------------
# Service factory (overridable in tests via patch)
# ---------------------------------------------------------------------------


def create_services() -> tuple[ScreenerService, WatchlistService, NewsService, SymbolService]:
    """Instantiate and wire all services."""
    provider = YahooProvider()
    store = ScreenerDataStore(base_path=DEFAULT_DATA_PATH)
    news = NewsService(provider=provider)
    symbol_svc = SymbolService(
        fetchers=build_fetchers(SYMBOL_SOURCES),
        cache_dir=DEFAULT_DATA_PATH,
        refresh_interval_hours=SYMBOL_REFRESH_INTERVAL_HOURS,
    )
    screener = ScreenerService(
        provider=provider, store=store, news_service=news, symbol_service=symbol_svc
    )
    watchlist = WatchlistService(store=store)
    return screener, watchlist, news, symbol_svc


# Module-level singletons (replaced in tests via patch)
_screener, _watchlist, _news, _symbol_svc = create_services()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_stock_screen(
    screen_type: str,
    criteria: dict,
    watchlist: str | None = None,
    save_result: str | None = None,
) -> dict:
    """Screen a list of stocks and return those that match all criteria.

    Args:
        screen_type: Screening strategy to apply. One of:
            - "technical"    — price, volume, RSI, moving averages, ATR
            - "fundamental"  — market cap, P/E, dividend yield, revenue growth, ETF metrics
            - "options"      — implied volatility, option volume, put/call ratio, earnings date
            - "news"         — keyword matching, management changes, date range
            - "custom"       — combine any of the above in a single pass

        criteria: Dict of filter thresholds. Keys depend on screen_type:

            TECHNICAL criteria (all optional):
              symbols (list[str])   — explicit list of symbols to screen
              min_price (float)     — reject if price < this value (e.g. 10.0)
              max_price (float)     — reject if price > this value (e.g. 500.0)
              min_volume (int)      — reject if avg daily volume < this value (e.g. 500000)
              min_rsi (float)       — reject if RSI(14) < this value (e.g. 30)
              max_rsi (float)       — reject if RSI(14) > this value (e.g. 70)
              above_sma_200 (bool)  — reject if price is below 200-day SMA
              above_sma_50 (bool)   — reject if price is below 50-day SMA
              max_atr_pct (float)   — reject if ATR% > this value (e.g. 5.0)
              category (str)        — default symbol pool: mega_cap|large_cap|mid_cap|small_cap|micro_cap|etf

            FUNDAMENTAL criteria (all optional):
              symbols (list[str])
              min_market_cap (int)      — e.g. 10_000_000_000 (10 B)
              min_pe / max_pe (float)   — forward P/E range, e.g. {"min_pe": 5, "max_pe": 30}
              min_dividend (float)      — minimum dividend yield in % (e.g. 2.0 means 2%)
              min_revenue_growth (float)— minimum YoY revenue growth as decimal (e.g. 0.05 = 5%)
              min_aum (int)             — ETF only: minimum assets under management
              max_expense_ratio (float) — ETF only: maximum expense ratio (e.g. 0.005)
              category (str)

            OPTIONS criteria (all optional):
              symbols (list[str])
              min_iv / max_iv (float)       — ATM implied volatility in % (e.g. {"min_iv": 20, "max_iv": 80})
              min_option_volume (int)        — minimum total call+put volume (e.g. 5000)
              min_put_call_ratio (float)     — minimum put/call volume ratio (e.g. 0.5)
              max_spread (float)             — maximum ATM bid-ask spread in % (e.g. 10.0)
              min_days_to_earnings (int)     — reject if earnings sooner than N days
              max_days_to_earnings (int)     — reject if earnings later than N days
              category (str)

            NEWS criteria (all optional):
              symbols (list[str])
              keywords (list[str])           — e.g. ["acquisition", "merger"]
              exclude_keywords (list[str])   — e.g. ["lawsuit", "recall"]
              require_all_keywords (bool)    — AND logic instead of OR (default false)
              min_days (int)                 — minimum age of news in days (default 0)
              max_days (int)                 — maximum age of news in days (default 30)
              management_changes (bool)      — require at least one management-change article

            CUSTOM criteria — nest sub-dicts under their type key:
              symbols (list[str])            — symbol list applies to all sub-screens
              technical (dict)               — same keys as TECHNICAL above
              fundamental (dict)             — same keys as FUNDAMENTAL above
              options (dict)                 — same keys as OPTIONS above
              news (dict)                    — same keys as NEWS above
              A symbol is rejected as soon as one sub-screen fails (short-circuit).

            Example — find cheap large-cap tech stocks with bullish technicals:
              screen_type = "custom"
              criteria = {
                  "symbols": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
                  "technical": {"above_sma_200": true, "max_rsi": 65},
                  "fundamental": {"min_market_cap": 100_000_000_000, "max_pe": 30}
              }

        watchlist: Name of a saved watchlist to use as the symbol source.
            Ignored when criteria already contains a "symbols" key.

        save_result: If provided, the result is persisted under this name and
            can be retrieved later with get_screening_result.

    Returns:
        Dict with keys: screen_type, criteria, matches (int), results (list),
        rejected (list with rejection_reasons), timestamp.
    """
    try:
        result = await _screener.run(
            screen_type=screen_type,
            criteria=criteria,
            watchlist_name=watchlist,
        )
        if save_result:
            _screener.store.save_screening_result(save_result, result)
        return result
    except (ValidationError, ValueError) as e:
        logger.error(f"Validation error in run_stock_screen: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in run_stock_screen: {e}")
        return {"error": f"Internal error: {e}"}


@mcp.tool()
async def get_stock_news(symbol: str, days_back: int = 30) -> dict:
    """Get recent news and company updates for a stock.

    Args:
        symbol: Stock ticker symbol (e.g. "AAPL", "MSFT").
        days_back: How many days of news history to retrieve (default 30).

    Returns:
        Dict with keys: recent_news, key_events, management_changes,
        current_management, company_info, last_updated.
    """
    try:
        return await _news.get_news_data(symbol, days_back=days_back)
    except Exception as e:
        logger.error(f"Error in get_stock_news for {symbol}: {e}")
        return {"error": str(e)}


@mcp.tool()
async def manage_watchlist(
    action: str,
    name: str,
    symbols: list[str] | None = None,
) -> dict:
    """Create, update, delete, or retrieve a named watchlist of stock symbols.

    Args:
        action: Operation to perform. One of:
            - "create" — create a new watchlist (requires symbols)
            - "update" — replace the symbols in an existing watchlist (requires symbols)
            - "delete" — remove the watchlist
            - "get"    — retrieve the symbol list

        name: Watchlist name. Must be 1-50 characters, alphanumeric with _ and -.
            Examples: "my_watchlist", "tech-picks", "sp500-subset"

        symbols: List of stock ticker symbols (required for create/update).
            Each symbol must be 1-10 characters. Maximum 1000 symbols.
            Example: ["AAPL", "MSFT", "GOOGL"]

    Returns:
        Dict with a "message" key on success, or "name"+"symbols" for get action.
    """
    try:
        return await _watchlist.dispatch(action, name, symbols)
    except (ValidationError, ValueError) as e:
        logger.error(f"Validation error in manage_watchlist: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in manage_watchlist: {e}")
        return {"error": f"Internal error: {e}"}


@mcp.tool()
async def get_screening_result(name: str) -> dict:
    """Retrieve a previously saved screening result.

    Args:
        name: The name used when the result was saved via run_stock_screen's
            save_result parameter.

    Returns:
        The full screening result dict, or {"error": "..."} if not found.
    """
    try:
        result = _screener.store.load_screening_result(name)
        if result is None:
            return {"error": f"Screening result '{name}' not found"}
        return result
    except Exception as e:
        logger.error(f"Error in get_screening_result for {name}: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@mcp.tool()
async def refresh_symbols(category: str | None = None) -> dict:
    """Force a refresh of the symbol lists fetched from index sources.

    Args:
        category: One of the registered index names (e.g. "cac40", "sp500",
            "nasdaq100", "sbf120", "dax", "ftse100", "aex").
            Pass None (omit the argument) to refresh all sources at once.

    Returns:
        Dict mapping each refreshed category to the number of symbols fetched,
        or to {"error": "..."} if the fetch failed for that category.
    """
    try:
        return await _symbol_svc.refresh(category)
    except (ValidationError, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in refresh_symbols: {e}")
        return {"error": f"Internal error: {e}"}


async def _startup() -> None:
    """Run once at server startup: seed missing/stale caches and launch background task."""
    if REFRESH_ON_STARTUP:
        logger.info("Startup symbol refresh…")
        result = await _symbol_svc.refresh()
        counts = {k: v for k, v in result.items() if not isinstance(v, dict)}
        errors = {k: v for k, v in result.items() if isinstance(v, dict)}
        logger.info(f"Startup refresh complete: {counts}")
        if errors:
            logger.warning(f"Startup refresh errors: {errors}")

    asyncio.create_task(
        _symbol_svc.start_background_refresh(),
        name="symbol-background-refresh",
    )
    logger.info(
        f"Background symbol refresh scheduled every {SYMBOL_REFRESH_INTERVAL_HOURS}h "
        f"(poll interval 1h)."
    )


def main() -> None:
    setup_logging()
    logger.info("Starting Stockscreen MCP server…")

    async def _run():
        await _startup()
        mcp.run()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
