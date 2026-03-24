"""FastMCP server — wires tools to services."""

import asyncio
import logging

from mcp.server.fastmcp import FastMCP

from stockscreen.config import (
    DEFAULT_DATA_PATH,
    EURONEXT_CACHE_TTL_SECONDS,
    PALMARES_CACHE_TTL_SECONDS,
    REFRESH_ON_STARTUP,
    SYMBOL_REFRESH_INTERVAL_HOURS,
    SYMBOL_SOURCES,
    setup_logging,
)
from stockscreen.exceptions import ValidationError
from stockscreen.providers.boursorama import BoursoramaProvider
from stockscreen.providers.euronext import EuronextProvider
from stockscreen.providers.facade import MarketDataFacade
from stockscreen.providers.symbol_fetchers.registry import build_fetchers
from stockscreen.providers.yahoo import YahooProvider
from stockscreen.providers.boursorama_palmares import BoursoramaPalmaresScaper
from stockscreen.services.news import NewsService
from stockscreen.services.palmares_service import PalmaresService
from stockscreen.services.screener import ScreenerService
from stockscreen.services.symbol_service import SymbolService
from stockscreen.services.watchlist import WatchlistService
from stockscreen.store.data_store import ScreenerDataStore
from stockscreen.store.palmares_store import PalmaresStore

logger = logging.getLogger("stockscreen-server-v1")

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP("stockscreen")

# ---------------------------------------------------------------------------
# Service factory (overridable in tests via patch)
# ---------------------------------------------------------------------------


def create_services() -> tuple[
    ScreenerService, WatchlistService, NewsService, SymbolService, PalmaresService
]:
    """Instantiate and wire all services."""
    yahoo = YahooProvider()
    boursorama = BoursoramaProvider(
        cache_dir=DEFAULT_DATA_PATH,
        cache_ttl_seconds=86400.0,
    )
    euronext = EuronextProvider(
        cache_dir=DEFAULT_DATA_PATH,
        cache_ttl_seconds=EURONEXT_CACHE_TTL_SECONDS,
    )
    facade = MarketDataFacade(yahoo=yahoo, boursorama=boursorama, euronext=euronext)

    store = ScreenerDataStore(base_path=DEFAULT_DATA_PATH)
    news = NewsService(provider=facade)
    symbol_svc = SymbolService(
        fetchers=build_fetchers(SYMBOL_SOURCES),
        cache_dir=DEFAULT_DATA_PATH,
        refresh_interval_hours=SYMBOL_REFRESH_INTERVAL_HOURS,
    )
    screener = ScreenerService(
        provider=facade, store=store, news_service=news, symbol_service=symbol_svc
    )
    watchlist = WatchlistService(store=store)
    palmares_svc = PalmaresService(
        scraper=BoursoramaPalmaresScaper(),
        store=PalmaresStore(base_path=DEFAULT_DATA_PATH),
        cache_ttl_seconds=PALMARES_CACHE_TTL_SECONDS,
    )
    return screener, watchlist, news, symbol_svc, palmares_svc


# Module-level singletons (replaced in tests via patch)
_screener, _watchlist, _news, _symbol_svc, _palmares_svc = create_services()


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
    """Get recent news and company updates for a single stock ticker.

    Use this tool to:
    - Surface M&A activity, earnings surprises, regulatory actions
    - Detect management changes (CEO/CFO/Chairman departures or appointments)
    - Find key events (SEC investigations, lawsuits, product launches)
    - Check news sentiment before screening options or fundamentals

    Args:
        symbol: Yahoo Finance ticker symbol. Use the exchange-suffixed form for
            non-US equities (e.g. "TTE.PA" for TotalEnergies Paris, "AIR.PA"
            for Airbus, "DBK.DE" for Deutsche Bank). US symbols need no suffix
            (e.g. "AAPL", "MSFT", "NVDA").

        days_back: Number of calendar days of history to include (default 30,
            max meaningful value ~90 — Yahoo rarely returns older articles).

    Returns:
        Dict with the following keys:

        recent_news (list[dict]):
            All news items within the date window, each with:
              title (str)           — article headline
              publisher (str)       — source outlet
              published (str)       — ISO datetime
              summary (str|None)    — article summary when available
              url (str|None)        — article URL

        key_events (list[dict]):
            Subset of recent_news matching regulatory / legal keywords
            (SEC, lawsuit, investigation, probe). Same structure as recent_news.

        management_changes (list[dict]):
            Subset matching executive-change keywords (CEO, CFO, chief,
            chairman, president). Same structure as recent_news.

        current_management (dict):
            Company officers from Yahoo: {name, title, totalPay} per officer.
            May be empty for ETFs and some foreign listings.

        company_info (dict):
            Static company metadata: sector, industry, country, website,
            longBusinessSummary, fullTimeEmployees, marketCap.

        last_updated (str):
            ISO datetime of when this response was generated.

    Example usage:
        get_stock_news(symbol="AAPL", days_back=14)
        get_stock_news(symbol="TTE.PA", days_back=30)
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

    Watchlists let you save a set of tickers under a memorable name and reuse
    them as the symbol pool for run_stock_screen without repeating the list
    every time.

    Typical workflow:
        1. manage_watchlist(action="create", name="cac-picks",
                            symbols=["TTE.PA", "AIR.PA", "MC.PA"])
        2. run_stock_screen(screen_type="fundamental",
                            criteria={"min_dividend": 3.0},
                            watchlist="cac-picks")

    Args:
        action: Operation to perform. Exactly one of:
            - "create" — create a new watchlist; fails if name already exists.
                         Requires symbols.
            - "update" — fully replace the symbol list of an existing watchlist.
                         Requires symbols.
            - "get"    — return the symbol list for the named watchlist.
            - "delete" — permanently remove the watchlist.

        name: Watchlist identifier. Rules:
            - 1 to 50 characters
            - Alphanumeric characters, underscore (_), and hyphen (-)
            - Cannot start with a hyphen
            - Examples: "tech-picks", "cac40_div", "my_watchlist"

        symbols: List of ticker symbols (required for "create" and "update").
            - Each symbol: 1–10 alphanumeric characters plus dot and hyphen
              (e.g. "AAPL", "TTE.PA", "BRK-B")
            - Maximum 1000 symbols per watchlist
            - Symbols are automatically uppercased

    Returns:
        - create/update/delete: {"message": "<confirmation text>"}
        - get:                  {"name": "<name>", "symbols": ["SYM1", ...]}
        - error:                {"error": "<description>"}

    Example usage:
        manage_watchlist(action="create", name="euronext-div",
                         symbols=["TTE.PA", "AIR.PA", "SAN.PA", "OR.PA"])
        manage_watchlist(action="get", name="euronext-div")
        manage_watchlist(action="update", name="euronext-div",
                         symbols=["TTE.PA", "MC.PA"])
        manage_watchlist(action="delete", name="euronext-div")
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
    """Retrieve a previously saved stock screening result by name.

    Results are saved by passing save_result="<name>" to run_stock_screen.
    Use this tool to re-examine a past screen without re-fetching market data,
    or to compare two runs of the same criteria at different dates.

    Args:
        name: The exact name used in the save_result parameter of run_stock_screen.
            Names follow the same rules as watchlist names (alphanumeric, _, -).

    Returns:
        The full screening result dict as originally returned by run_stock_screen,
        including: screen_type, criteria, matches, results, rejected, timestamp.
        Returns {"error": "Screening result '<name>' not found"} if absent.

    Example usage:
        # Save during screening:
        run_stock_screen(screen_type="fundamental",
                         criteria={"category": "cac40", "min_dividend": 3.0},
                         save_result="cac40-div-2026-03")
        # Retrieve later:
        get_screening_result(name="cac40-div-2026-03")
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
    """Force a refresh of the cached symbol lists for one or all index categories.

    Symbol lists are automatically fetched from Wikipedia and cached locally on
    first use. They expire after 24 hours (configurable via
    STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS). Use this tool when:
    - You want to pull in a recent index constituent change immediately.
    - A screening run returned fewer symbols than expected.
    - You have just changed STOCKSCREEN_SYMBOL_SOURCES.

    Args:
        category: Index source to refresh. Supported values:
            - "sp500"      — S&P 500 (≈ 503 US large-cap stocks)
            - "nasdaq100"  — Nasdaq 100 (≈ 102 US tech/growth stocks)
            - "cac40"      — CAC 40 (40 French blue-chips, ".PA" suffix)
            - "sbf120"     — SBF 120 (120 Euronext Paris stocks, ".PA" suffix)
            - "dax"        — DAX (40 German blue-chips, ".DE" suffix)
            - "ftse100"    — FTSE 100 (100 UK blue-chips, ".L" suffix)
            - "aex"        — AEX (25 Amsterdam stocks, ".AS" suffix)
            Pass None or omit to refresh ALL active sources at once.

    Returns:
        Dict mapping each category to the count of symbols fetched:
            {"sp500": 503, "cac40": 40, ...}
        On partial failure, the failing category maps to an error dict:
            {"sp500": 503, "dax": {"error": "HTTP 503"}}

    Example usage:
        refresh_symbols()                  # refresh all active sources
        refresh_symbols(category="cac40")  # refresh CAC 40 only
    """
    try:
        return await _symbol_svc.refresh(category)
    except (ValidationError, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in refresh_symbols: {e}")
        return {"error": f"Internal error: {e}"}


@mcp.tool()
async def get_palmares(
    min_rendement: float | None = None,
    max_rendement: float | None = None,
    nom_contains: str | None = None,
    limit: int = 50,
    force_refresh: bool = False,
) -> dict:
    """Get the Boursorama dividend palmares — French equities ranked by dividend yield.

    The palmares is scraped from Boursorama's multi-page dividend ranking table
    (https://www.boursorama.com/bourse/actions/palmares/dividendes/).  It covers
    French/Euronext-Paris equities and includes up to three years of dividend data.
    Results are sorted by best rendement descending (highest yield first), with
    entries that have no dividend data pushed to the bottom.

    The snapshot is cached on disk for 24 hours by default (configurable via
    STOCKSCREEN_PALMARES_CACHE_TTL env var in seconds). A cache hit returns
    immediately without any network call.

    Args:
        min_rendement:  Keep only entries whose best historical rendement ≥ this
            value (in %).  Example: 3.0 keeps only stocks yielding ≥ 3 %.

        max_rendement:  Keep only entries whose best historical rendement ≤ this
            value (in %).  Example: 10.0 excludes unusually high-yield/distressed
            stocks.

        nom_contains:   Case-insensitive substring match on the company name
            (nom field).  Example: "total" matches "TotalEnergies SE" and
            "Total Gabon".

        limit:          Maximum number of entries to return after filtering
            (default 50, capped to 500).  The total_entries field always reflects
            the unfiltered snapshot size.

        force_refresh:  Set to true to bypass the cache and trigger a fresh scrape
            immediately.  Useful when you suspect stale data or want today's
            dividend announcements.  Filters are still applied after the refresh.

    Returns:
        Dict with the following keys:

        fetched_at (str):
            ISO datetime of the cached snapshot (when the last scrape ran).

        total_entries (int):
            Total number of entries in the snapshot *before* any filtering.
            Useful to know how many stocks are in the palmares overall.

        returned (int):
            Number of entries actually returned after filtering and limit.

        entries (list[dict]):
            Filtered list of palmares entries, each with:
              code_bourso (str)     — Boursorama internal ticker (e.g. "1rTTE")
              nom (str)             — Company name (e.g. "TotalEnergies SE")
              cours (float|None)    — Last known price in EUR
              isin (str|None)       — ISIN if resolved, else None
              dividendes (list[dict]):
                Each element represents one dividend year:
                  annee (str)           — Year (e.g. "2025")
                  dividende (float|None)— Gross annual dividend in EUR
                  rendement (float|None)— Dividend yield in % (dividende/cours × 100)

    Example usage:
        # Top 20 stocks yielding between 3 % and 8 %:
        get_palmares(min_rendement=3.0, max_rendement=8.0, limit=20)

        # All "Total" group stocks in the palmares:
        get_palmares(nom_contains="total")

        # Force a fresh scrape then return top 50:
        get_palmares(force_refresh=True, limit=50)

    Typical workflow with run_stock_screen:
        # 1. Get high-yield candidates from palmares
        palmares = get_palmares(min_rendement=5.0, limit=30)
        symbols = [e["code_bourso"] for e in palmares["entries"]]

        # 2. Cross-check technicals on those candidates
        run_stock_screen(
            screen_type="technical",
            criteria={"symbols": symbols, "above_sma_200": true}
        )
    """
    try:
        limit = max(1, min(limit, 500))
        if force_refresh:
            await _palmares_svc.refresh()   # seed fresh cache, then fall through to get()
        snap = await _palmares_svc.get(
            min_rendement=min_rendement,
            max_rendement=max_rendement,
            nom_contains=nom_contains,
            limit=limit,
        )
        from dataclasses import asdict
        return {
            "fetched_at": snap.fetched_at,
            "total_entries": snap.total_entries,
            "returned": len(snap.entries),
            "entries": [asdict(e) for e in snap.entries],
        }
    except Exception as e:
        logger.error(f"Unexpected error in get_palmares: {e}")
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
