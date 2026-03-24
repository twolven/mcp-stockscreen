# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockScreen is a modular Python MCP (Model Context Protocol) server that provides stock screening tools to Claude Desktop. It fetches real-time market data via Yahoo Finance (`yfinance`) and exposes four MCP tools: `run_stock_screen`, `get_stock_news`, `manage_watchlist`, and `get_screening_result`.

## Running

```bash
pip install -r requirements.txt
python -m stockscreen.server
# or via entry point after pip install -e .
stockscreen
```

The server communicates via stdio-based MCP protocol — it's designed to be launched by Claude Desktop, not run interactively.

## Architecture

```
stockscreen/
├── __init__.py            # Public exports
├── server.py              # FastMCP tools (thin wrappers over services)
├── config.py              # Paths, logging, constants, legacy migration
├── exceptions.py          # StockscreenError, ValidationError, APIError
├── providers/
│   └── yahoo.py           # YahooProvider — only file that imports yfinance
├── models/
│   └── schemas.py         # Pydantic v2 models + StockscreenJSONEncoder
├── services/
│   ├── screener.py        # ScreenerService (technical, fundamental, options, custom)
│   ├── news.py            # NewsService
│   └── watchlist.py       # WatchlistService
└── store/
    └── data_store.py      # ScreenerDataStore + DefaultSymbols
```

### Key components

- **`YahooProvider`** — Wraps all `yf.Ticker` calls via `run_in_executor` (true async). Includes exponential backoff retry. The **only** file that imports `yfinance`.
- **`ScreenerDataStore`** — JSON file-based persistence for watchlists and screening results under `data/`. `DefaultSymbols` manages cached S&P 500/Dow/Nasdaq symbol lists categorised by market cap.
- **`ScreenerService`** — Unified screener with a single `run(screen_type, criteria, symbols, watchlist_name)` method dispatching to `_run_technical`, `_run_fundamental`, `_run_options`, `_run_news`, `_run_custom`.
- **`WatchlistService`** — CRUD operations on named watchlists with Pydantic validation.
- **`NewsService`** — News retrieval, categorisation, and keyword/date filtering.
- **FastMCP server (`server.py`)** — `FastMCP("stockscreen")` with four `@mcp.tool()` decorated functions. Module-level service singletons (`_screener`, `_watchlist`, `_news`) wired via `create_services()`.

## Data Flow

```
Claude Desktop → FastMCP stdio → @mcp.tool() → Service → YahooProvider → yfinance → pandas → JSON
```

## Key Details

- Data stored in `data/` relative to the package (screening_results/, watchlists/), auto-created on first run
- `STOCKSCREEN_DATA_PATH` env var overrides the default data path
- Retry with exponential backoff (3 attempts) lives in `YahooProvider`, not in the server layer
- `StockscreenJSONEncoder` handles pandas Timestamp/NaT/Period and numpy types for JSON serialisation
- Symbol validation caps at 1000 symbols per request (Pydantic model `StockSymbols`)
- Watchlist names: 1-50 chars, alphanumeric plus `_` and `-` (Pydantic model `WatchlistName`)
- `services/` never imports `mcp` — services are testable without a running server
- `server.py` never imports `yfinance` — decoupled from the data layer

## Testing

```bash
source venv/bin/activate
pytest            # runs all tests
pytest tests/test_screener_service.py   # specific module
```

Test suite: ~152 tests covering all modules with async pytest (`asyncio_mode = "auto"`). All provider calls are mocked at the `YahooProvider` level — no real network calls.
