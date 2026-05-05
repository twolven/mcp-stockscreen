# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockScreen is a modular Python MCP (Model Context Protocol) server that provides stock screening tools to Claude Desktop / Claude Code. It exposes six MCP tools: `run_stock_screen`, `get_stock_news`, `manage_watchlist`, `get_screening_result`, `refresh_symbols`, and `get_palmares`.

Market data is fetched from three sources via a unified `MarketDataFacade`:
- **Yahoo Finance** (`yfinance`) — price, volume, RSI, P/E, options, news
- **Boursorama** (scraped) — dividends, rendement, consensus, performance history for Euronext Paris
- **Euronext** (scraped) — bidirectional ISIN ↔ Yahoo ticker resolution

A separate `BoursoramaPalmaresScaper` feeds the `get_palmares` tool from the Boursorama dividend ranking table.

## Running

```bash
uv sync
uv run stockscreen
# or
uv run python -m stockscreen.server
```

The server communicates via stdio-based MCP protocol — it's designed to be launched by Claude Desktop, not run interactively.

## Architecture

```
stockscreen/
├── server.py                          # FastMCP tools (thin wrappers over services)
├── config.py                          # Paths, logging, env overrides, constants
├── exceptions.py                      # StockscreenError, ValidationError, APIError
├── providers/
│   ├── yahoo.py                       # YahooProvider — only file that imports yfinance
│   ├── boursorama.py                  # BoursoramaProvider — Boursorama scraper (async)
│   ├── euronext.py                    # EuronextProvider — bidirectional ISIN↔ticker
│   ├── facade.py                      # MarketDataFacade — single entry point for all providers
│   ├── boursorama_palmares.py         # BoursoramaPalmaresScaper — multi-page dividend ranking
│   └── symbol_fetchers/
│       ├── base.py                    # BaseSymbolFetcher ABC + SymbolRecord
│       ├── wikipedia.py               # SP500, Nasdaq100, CAC40, SBF120, DAX, FTSE100, AEX
│       └── registry.py               # build_fetchers(["cac40", "sp500", ...])
├── models/schemas.py                  # Pydantic v2 validation + StockscreenJSONEncoder
├── services/
│   ├── screener.py                    # ScreenerService (technical/fundamental/options/news/custom)
│   ├── news.py                        # NewsService
│   ├── watchlist.py                   # WatchlistService
│   ├── symbol_service.py              # SymbolService — fetch/cache/refresh index symbol lists
│   └── palmares_service.py            # PalmaresService — cache, filter, sort palmares snapshot
└── store/
    ├── data_store.py                  # ScreenerDataStore + DefaultSymbols (JSON persistence)
    └── palmares_store.py              # PalmaresStore — palmares snapshot persistence
```

### Key components

- **`YahooProvider`** — Wraps all `yf.Ticker` calls via `run_in_executor` (true async). Includes exponential backoff retry (3 attempts). The **only** file that imports `yfinance`.

- **`BoursoramaProvider`** — Scrapes Boursorama for Euronext Paris data (cours, dividende, rendement, last_dividend_date, consensus, performance). Accepts ISIN or Boursorama code. One JSON cache file per ticker. Stale fallback on network error.

- **`EuronextProvider`** — Resolves identifiers bidirectionally:
  - `resolve_ticker(isin)` → `EuronextRecord` (with `yahoo_ticker` like `TTE.PA`)
  - `resolve_isin(ticker)` → `EuronextRecord` (with `isin` like `FR0000131104`)
  - Cache is shared: resolving a ticker also writes the ISIN-keyed cache file (and vice versa).
  - Supports 11 MIC → Yahoo suffix mappings: XPAR→.PA, XETR→.DE, XLON→.L, XAMS→.AS, XMIL→.MI, XMAD→.MC, XBRU→.BR, XLIS→.LS, XHEL→.HE, XSTO→.ST, XOSL→.OL

- **`MarketDataFacade`** — Single entry point for all data. Accepts a ticker (e.g. `TTE.PA`) or an ISIN (12-char string starting with 2 alpha chars). For ISINs, calls `EuronextProvider.resolve_ticker` first. Calls Yahoo and Boursorama in parallel via `asyncio.gather`. Merges results: **Boursorama-first** for `dividende`, `rendement`, `last_dividend_date`, `consensus`, `performance`; Yahoo provides everything else. The facade is what `ScreenerService` and `NewsService` receive as their `provider`.

- **`BoursoramaPalmaresScaper`** — Scrapes the full multi-page Boursorama dividend ranking table (`/bourse/actions/palmares/dividendes/page-{N}`). Extracts up to 3 years of dividend data per stock dynamically from column headers. Returns `list[PalmaresEntry]`.

- **`PalmaresService`** — Orchestrates scraping, caching, sorting, and filtering. `get()` returns a `PalmaresSnapshot` sorted by best rendement descending (None last), with `total_entries` reflecting the unfiltered count. `refresh()` forces a new scrape regardless of cache freshness.

- **`ScreenerService`** — Unified screener. Single `run(screen_type, criteria, symbols, watchlist_name)` method dispatching to `_run_technical`, `_run_fundamental`, `_run_options`, `_run_news`, `_run_custom`. Provider injected via constructor — no coupling to Yahoo or Boursorama directly.

- **`WatchlistService`** — CRUD operations on named watchlists with Pydantic validation.

- **`NewsService`** — News retrieval, categorisation, and keyword/date filtering.

- **`SymbolService`** — Fetches and caches index constituent lists (CAC 40, S&P 500…) from Wikipedia via pluggable `BaseSymbolFetcher` implementations. Background refresh loop every N hours.

- **`ScreenerDataStore`** — JSON file-based persistence for watchlists and screening results. `DefaultSymbols` manages cached symbol lists categorised by market cap.

- **`PalmaresStore`** — Reads/writes a single `PalmaresSnapshot` JSON file at `{data}/palmares/palmares_dividendes.json`.

- **FastMCP server (`server.py`)** — `FastMCP("stockscreen")` with six `@mcp.tool()` decorated async functions. Module-level service singletons wired via `create_services()` which returns a 5-tuple `(screener, watchlist, news, symbol_svc, palmares_svc)`.

## Data Flow

```
Claude Desktop
  → FastMCP stdio
    → @mcp.tool() (server.py)
      → Service
        → MarketDataFacade
          ├── YahooProvider  → yfinance → pandas
          ├── BoursoramaProvider → HTTP scrape → merge (Boursorama-first for dividends)
          └── EuronextProvider → HTTP → ISIN↔ticker cache
        → BoursoramaPalmaresScaper → HTTP multi-page → PalmaresStore (JSON)
        → SymbolService → Wikipedia fetchers → JSON cache
        → ScreenerDataStore → JSON files (watchlists, screening results)
```

## Key Details

- Data stored under `STOCKSCREEN_DATA_PATH` (default: `stockscreen/data/`), auto-created on first run
- `STOCKSCREEN_DATA_PATH` env var overrides the default
- Retry with exponential backoff (3 attempts) lives in `YahooProvider`, not in the server layer
- `StockscreenJSONEncoder` handles pandas Timestamp/NaT/Period and numpy types for JSON serialisation
- Symbol validation caps at 1000 symbols per request (Pydantic model `StockSymbols`)
- Watchlist names: 1–50 chars, alphanumeric plus `_` and `-` (Pydantic model `WatchlistName`)
- `services/` never imports `mcp` — services are testable without a running server
- `server.py` never imports `yfinance` — decoupled from the data layer
- `ScreenerService` and `NewsService` accept `MarketDataFacade` as their provider (duck typing — same interface as `YahooProvider`)
- ISIN detection: 12-char string where first 2 chars are alpha → treated as ISIN by `MarketDataFacade`
- Palmares `total_entries` always reflects the unfiltered count; `returned` is the post-filter count
- `force_refresh=True` in `get_palmares`: calls `refresh()` to seed the cache, then always calls `get()` to apply filters

## Testing

```bash
uv run pytest            # runs all tests
uv run pytest tests/test_screener_service.py   # specific module
```

Test suite: ~290 tests covering all modules with async pytest (`asyncio_mode = "auto"`). All provider calls are mocked — no real network calls in the test suite.

Key test files:
- `tests/test_server.py` — FastMCP tool routing tests
- `tests/test_screener_service.py` — ScreenerService unit tests
- `tests/test_boursorama_provider.py` — BoursoramaProvider scraping + cache tests
- `tests/test_euronext_provider.py` — EuronextProvider bidirectional resolution tests
- `tests/test_market_data_facade.py` — MarketDataFacade merge/fallback tests
- `tests/test_boursorama_palmares.py` — Palmares scraper HTML parsing tests
- `tests/test_palmares_store.py` — PalmaresStore read/write tests
- `tests/test_palmares_service.py` — PalmaresService cache/filter/sort tests
