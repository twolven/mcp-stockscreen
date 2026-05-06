# AGENTS.md

StockScreen — Python MCP server exposing 6 stock screening tools via FastMCP (supports stdio, SSE, and Streamable HTTP transports).

## Commands

```bash
uv sync                   # install deps (no pip — uv only)
uv run stockscreen        # launch server
uv run python -m stockscreen.server  # alternative launch

# Testing
uv run pytest                                    # ~290 tests, async, no network
uv run pytest tests/test_screener_service.py     # single module
uv run pytest --cov=stockscreen --cov-report=term-missing  # with coverage
```

## Architecture rules agents MUST follow

- **Language**: Chat communication is in French. Code, documentation, comments, and file names are in English.
- **`services/` never imports `mcp`** — services are testable standalone.
- **`server.py` never imports `yfinance`** — only `providers/yahoo.py` does.
- **`provider=` parameter in services** is always a `MarketDataFacade` (duck-typed as `YahooProvider`).
- **DI factory**: `server.py:create_services()` wires all singletons; tests patch it to inject mocks.
- **Docs update**: After any code change, update `README.md` and `ARCHITECTURE.md` to reflect the new state.

## Key runtime facts

| Fact | Detail |
|---|---|
| Package manager | `uv` only — no pip, no poetry |
| Async test mode | `asyncio_mode = "auto"` (in pyproject.toml, no decorators needed) |
| Test isolation | `conftest.py` sets `STOCKSCREEN_DATA_PATH` to a temp dir **before any import** — critical ordering |
| Transport | Multi-protocol — `stdio` (default), `sse`, `streamable-http`. Config via `STOCKSCREEN_TRANSPORT` env var. HTTP host/port via `STOCKSCREEN_HOST`/`STOCKSCREEN_PORT`. |
| Data dir | `stockscreen/data/` (override via `STOCKSCREEN_DATA_PATH` env var) |
| ISIN detection | 12-char string where first 2 chars are alpha → treated as ISIN by facade |
| Boursorama merge | Fields `dividende`, `rendement`, `last_dividend_date`, `consensus`, `performance` come from Boursorama first, Yahoo as fallback |
| Euronext cache | Bidirectional — resolving ISIN→ticker also writes the ISIN-keyed cache file, and vice versa |
| Yahoo async | All `yf.Ticker` calls go through `loop.run_in_executor(None, ...)` with exponential backoff (3 retries, 1s/2s/4s) |
| Palmares | `total_entries` = unfiltered count; `returned` = post-filter. `force_refresh=True` calls `refresh()` then `get()` for filtering. |
| Symbol limit | Max 1000 symbols per request (Pydantic `StockSymbols`) |
| Watchlist names | 1–50 chars, `[a-zA-Z0-9_][a-zA-Z0-9_-]*` (Pydantic `WatchlistName`) |
| Logging | Logger name `"stockscreen-server-v1"`, shared across all modules |
| JSON encoder | `StockscreenJSONEncoder` handles pandas Timestamp/NaT/Period and numpy types |

## Project structure

```
stockscreen/
  server.py                    # FastMCP tools + create_services() DI factory
  config.py                    # env vars, paths, logging
  exceptions.py                # StockscreenError, ValidationError, APIError
  providers/
    yahoo.py                   # YahooProvider — ONLY file importing yfinance
    boursorama.py              # BoursoramaProvider — Euronext Paris scraper
    euronext.py                # EuronextProvider — ISIN↔ticker (11 MICs)
    facade.py                  # MarketDataFacade — single entry point, ISIN auto-detect
    boursorama_palmares.py     # BoursoramaPalmaresScaper — multi-page dividend ranking
    symbol_fetchers/           # Wikipedia-based index fetchers (SP500, CAC40, ...)
  services/                    # Business logic — NEVER imports mcp or yfinance
    screener.py, news.py, watchlist.py, symbol_service.py, palmares_service.py
  store/                       # JSON persistence
    data_store.py, palmares_store.py
  models/schemas.py            # Pydantic v2 validation + StockscreenJSONEncoder
```

Refer to `ARCHITECTURE.md` for class contracts, data flow diagrams, and exception hierarchy.
