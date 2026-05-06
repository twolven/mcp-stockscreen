# StockScreen MCP — Software Architecture

> Quick-reference for Claude Code. Describes modules, patterns, class contracts,
> and wiring so you can locate any concept in one read.

---

## 1. System Overview

StockScreen is a **FastMCP server** exposing six tools to Claude Desktop / Claude Code and other MCP-compatible clients.
It supports three transport protocols (stdio, SSE, Streamable HTTP) and aggregates market data from three external sources, persisting results locally as JSON files.

```
Client (Claude Desktop / Claude Code / HTTP)
  │  MCP (stdio / SSE / Streamable HTTP)
  ▼
FastMCP  (mcp.server.fastmcp.FastMCP)
  │
  ├── 6 @mcp.tool() functions  →  server.py
  │
  ├── Services layer           →  stockscreen/services/
  │     ScreenerService, NewsService, WatchlistService,
  │     SymbolService, PalmaresService
  │
  ├── Providers layer          →  stockscreen/providers/
  │     MarketDataFacade
  │       ├── YahooProvider      (yfinance, async via run_in_executor)
  │       ├── BoursoramaProvider (HTTP scrape, async via run_in_executor)
  │       └── EuronextProvider   (REST API, async via run_in_executor)
  │     BoursoramaPalmaresScaper (multi-page scrape)
  │     BaseSymbolFetcher + Wikipedia implementations
  │
  └── Store layer              →  stockscreen/store/
        ScreenerDataStore, PalmaresStore
```

**Transport**: Multi-protocol — `stdio` (default), `sse` (SSE), or `streamable-http` (modern HTTP).
Configured via `STOCKSCREEN_TRANSPORT` env var in `config.py`.
Host/port for HTTP transports via `STOCKSCREEN_HOST` / `STOCKSCREEN_PORT`.
**Entry point**: `stockscreen/server.py :: main()` → `asyncio.run(_run())` dispatches to `run_stdio_async()`, `run_sse_async()`, or `run_streamable_http_async()`.

---

## 2. Package Layout

```
stockscreen/
├── __init__.py
├── server.py                        # FastMCP wiring + 6 tools + create_services()
├── config.py                        # Env vars, paths, logging setup
├── exceptions.py                    # StockscreenError hierarchy
│
├── models/
│   └── schemas.py                   # Pydantic models + StockscreenJSONEncoder
│
├── providers/
│   ├── yahoo.py                     # YahooProvider — ONLY file that imports yfinance
│   ├── boursorama.py                # BoursoramaProvider — Euronext Paris scraper
│   ├── euronext.py                  # EuronextProvider — ISIN↔ticker resolution
│   ├── facade.py                    # MarketDataFacade — single entry point
│   ├── boursorama_palmares.py       # BoursoramaPalmaresScaper + PalmaresEntry
│   └── symbol_fetchers/
│       ├── base.py                  # BaseSymbolFetcher (ABC) + SymbolRecord
│       ├── wikipedia.py             # 7 concrete fetchers (SP500, CAC40, ...)
│       └── registry.py             # build_fetchers(["cac40", "sp500"])
│
├── services/
│   ├── screener.py                  # ScreenerService — technical/fundamental/options/news/custom
│   ├── news.py                      # NewsService — fetch + categorise + screen
│   ├── watchlist.py                 # WatchlistService — CRUD via ScreenerDataStore
│   ├── symbol_service.py            # SymbolService — fetch/cache/background refresh
│   └── palmares_service.py          # PalmaresService — cache + filter + sort
│
└── store/
    ├── data_store.py                # ScreenerDataStore + DefaultSymbols
    └── palmares_store.py            # PalmaresStore + PalmaresSnapshot
```

---

## 3. Layers and Dependencies

```
server.py
  └─ depends on ──► services/*   (via create_services DI factory)
                    models/schemas.py
                    exceptions.py
                    config.py

services/*
  └─ depends on ──► providers/facade.py  (MarketDataFacade)
                    store/data_store.py  (ScreenerDataStore)
                    exceptions.py
  NEVER imports ──► mcp   (services are testable standalone)
  NEVER imports ──► yfinance

providers/facade.py
  └─ depends on ──► providers/yahoo.py
                    providers/boursorama.py
                    providers/euronext.py

providers/yahoo.py
  └─ depends on ──► yfinance  (ONLY here)

providers/boursorama.py, euronext.py
  └─ depend on ──► requests (sync, always called via run_in_executor)

providers/boursorama_palmares.py
  └─ depends on ──► requests, lxml
```

**Isolation rule**: `services/` never imports `mcp`; `server.py` never imports `yfinance`.

---

## 4. Exception Hierarchy

File: `stockscreen/exceptions.py`

```python
StockscreenError(Exception)
  ├── ValidationError   # bad input (caught at tool boundary → {"error": ...})
  └── APIError          # provider failure (ISIN not resolvable, HTTP error)
```

`server.py` catches `(ValidationError, ValueError)` per tool and returns `{"error": str(e)}`.
Unexpected exceptions return `{"error": f"Internal error: {e}"}`.

---

## 5. Key Classes

### 5.1 Providers

#### `YahooProvider` — `providers/yahoo.py`

- **Only file** that imports `yfinance`.
- All methods are `async`; blocking `yf.Ticker` calls go through `loop.run_in_executor(None, ...)`.
- Decorated with `@_retry(max_retries=3, delay=1.0)` — exponential backoff (1s, 2s, 4s).
- Public API:
  ```python
  get_ticker_info(symbol) -> dict | None
  get_history(symbol, period="1y") -> pd.DataFrame
  get_option_chain(symbol, expiry) -> Any
  get_option_expirations(symbol) -> tuple
  get_news(symbol) -> list[dict]
  get_earnings_dates(symbol) -> dict   # {next_earnings, days_to_earnings, ...}
  ```

#### `BoursoramaProvider` — `providers/boursorama.py`

- Scrapes Boursorama for Euronext Paris data (dividende, rendement, consensus, performance).
- Accepts ISIN or Boursorama code as query.
- One JSON cache file per ticker under `cache_dir/`.
- Stale fallback on network error.

#### `EuronextProvider` — `providers/euronext.py`

- Bidirectional resolution: ISIN → Yahoo ticker and ticker → ISIN.
- Uses Euronext Live REST API (`live.euronext.com`).
- Cache: `{cache_dir}/euronext_{key}.json` (one file per ISIN or normalised ticker).
- Resolving either direction writes **both** keyed files (shared cache).
- MIC → Yahoo suffix mapping (11 exchanges): `XPAR→.PA`, `XETR→.DE`, `XLON→.L`, etc.
- Key dataclass:
  ```python
  @dataclass
  class EuronextRecord:
      isin: str
      symbol: str        # "TTE"
      name: str
      mic: str           # "XPAR"
      yahoo_ticker: str  # "TTE.PA"
      cached_at: str
  ```
- Helper exposed for import: `_normalize_ticker(ticker) -> str` (strips `.PA`, `.DE`, …).

#### `MarketDataFacade` — `providers/facade.py`

- **Single entry point** used by all services. Duck-type compatible with `YahooProvider`.
- Accepts Yahoo ticker (`"TTE.PA"`) **or** ISIN (`"FR0000131104"`).
- ISIN detection: `len == 12 and first 2 chars are alpha`.
- For ISINs: calls `EuronextProvider.resolve_ticker` first → raises `APIError` if unresolvable.
- Yahoo and Boursorama calls are fired **in parallel** via `asyncio.gather`.
- Merge strategy: **Boursorama-first** for `dividende`, `rendement`, `last_dividend_date`, `consensus`, `performance`; Yahoo fallback for all others.
- Public API (same contract as `YahooProvider` + extras):
  ```python
  get_quote(identifier) -> dict           # enriched merged dict
  get_ticker_info(identifier) -> dict     # alias for get_quote
  get_history(identifier, period) -> pd.DataFrame
  get_news(identifier) -> list[dict]
  get_option_chain(identifier, expiry) -> Any
  get_option_expirations(identifier) -> tuple
  get_earnings_dates(identifier) -> dict
  ```

#### `BoursoramaPalmaresScaper` — `providers/boursorama_palmares.py`

- Scrapes `boursorama.com/bourse/actions/palmares/dividendes/page-{N}`.
- Auto-detects total page count from pagination HTML.
- HTTP done via `requests.Session` (sync) wrapped in `run_in_executor`.
- Year columns extracted dynamically from `<th>` headers.
- Key dataclass:
  ```python
  @dataclass
  class PalmaresEntry:
      code_bourso: str
      nom: str
      cours: float | None
      dividendes: list[dict]  # [{annee, dividende, rendement}, ...]
      isin: str | None
  ```
- Public API:
  ```python
  fetch_all() -> list[PalmaresEntry]
  fetch_page(page: int) -> list[PalmaresEntry]
  ```

#### `BaseSymbolFetcher` — `providers/symbol_fetchers/base.py`

Abstract base for all index fetchers. Concrete subclasses must define:
```python
class MyFetcher(BaseSymbolFetcher):
    name: ClassVar[str] = "my_index"       # used as cache key
    source_url: ClassVar[str] = "https://..."

    async def fetch(self) -> list[SymbolRecord]: ...
```

`SymbolRecord` dataclass:
```python
@dataclass
class SymbolRecord:
    symbol: str            # Yahoo ticker, uppercased
    name: str
    market_cap: float | None
    instrument_type: str | None   # "equity" | "etf"
```

Registered fetchers (`providers/symbol_fetchers/registry.py`):
`sp500`, `nasdaq100`, `cac40`, `sbf120`, `dax`, `ftse100`, `aex`
All implemented in `wikipedia.py` via Wikipedia HTML table scraping.

---

### 5.2 Services

#### `ScreenerService` — `services/screener.py`

Constructor injection (DI):
```python
ScreenerService(
    provider: MarketDataFacade,
    store: ScreenerDataStore,
    news_service: NewsService,
    symbol_service: SymbolService | None,
)
```

Single public method:
```python
async def run(screen_type, criteria, symbols=None, watchlist_name=None) -> dict
```

Symbol resolution priority (first match wins):
1. `symbols` parameter
2. `watchlist_name` → `store.load_watchlist()`
3. `criteria["symbols"]`
4. `criteria["category"]` → `symbol_service.get()`
5. Empty list

Dispatches to private methods:
- `_run_technical` — price, volume, SMA20/50/200, RSI(14), ATR(14)
- `_run_fundamental` — market cap, P/E, dividend yield, revenue growth; ETF branch (AUM, expense ratio)
- `_run_options` — IV, option volume, put/call ratio, bid-ask spread, earnings date
- `_run_news` → delegates to `news_service.screen_news()`
- `_run_custom` — composes technical + fundamental + options + news per symbol, short-circuit on first rejection

All screen methods return:
```python
{
  "screen_type": str,
  "criteria": dict,
  "matches": int,
  "results": list,     # passed symbols
  "rejected": list,    # with rejection_reasons
  "timestamp": str,    # ISO datetime
}
```

#### `NewsService` — `services/news.py`

```python
NewsService(provider: MarketDataFacade)
```

- `get_news_data(symbol, days_back=30) -> dict` — fetches and categorises into `recent_news`, `key_events`, `management_changes`.
- `screen_news(symbols, criteria) -> dict` — filters by keywords, date range, management changes.
- Classification keywords:
  - Management: `ceo`, `chief`, `executive`, `president`, `chairman`
  - Key events: `lawsuit`, `investigation`, `sec`, `probe`

#### `WatchlistService` — `services/watchlist.py`

```python
WatchlistService(store: ScreenerDataStore)
```

CRUD via `dispatch(action, name, symbols)` → routes to `create / get / update / delete`.
Validates with `WatchlistName` and `StockSymbols` Pydantic models before touching the store.

#### `SymbolService` — `services/symbol_service.py`

```python
SymbolService(fetchers: list[BaseSymbolFetcher], cache_dir, refresh_interval_hours=24)
```

- `get(category) -> list[str]` — returns cached symbols, fetches if stale, falls back to stale on error.
- `refresh(category=None) -> dict` — force-fetches one or all categories.
- `start_background_refresh(poll_interval=3600)` — coroutine launched as asyncio task at startup; wakes every hour, refreshes any expired category.
- Cache: `{cache_dir}/{category}.json` with `{timestamp, symbols: [{symbol, name, market_cap, ...}]}`.

#### `PalmaresService` — `services/palmares_service.py`

```python
PalmaresService(scraper: BoursoramaPalmaresScaper, store: PalmaresStore, cache_ttl_seconds=86400)
```

- `get(min_rendement, max_rendement, nom_contains, limit) -> PalmaresSnapshot` — loads or fetches, sorts by best rendement descending (None last), filters, slices.
- `refresh() -> PalmaresSnapshot` — bypasses cache, scrapes fresh, writes to store.
- `total_entries` in the returned snapshot always reflects the **unfiltered** count.

---

### 5.3 Store

#### `ScreenerDataStore` — `store/data_store.py`

JSON file store. Creates `{base_path}/screening_results/`, `watchlists/`, `market_data/` on init.

```python
save_watchlist(name, symbols)   → {base_path}/watchlists/{name}.json
load_watchlist(name) -> list | None
delete_watchlist(name) -> bool
save_screening_result(name, data)  → uses StockscreenJSONEncoder
load_screening_result(name) -> dict | None
```

Also owns a `DefaultSymbols` instance (legacy category fallback — not wired in current version).

#### `PalmaresStore` — `store/palmares_store.py`

Single-file store for the dividend palmares snapshot.

```python
save(snapshot: PalmaresSnapshot) -> None
load() -> PalmaresSnapshot | None
```

File path: `{base_path}/palmares/palmares_dividendes.json`

`PalmaresSnapshot` dataclass:
```python
@dataclass
class PalmaresSnapshot:
    fetched_at: str        # ISO datetime
    page_count: int
    total_entries: int
    entries: list[PalmaresEntry]
```

---

### 5.4 Models

File: `stockscreen/models/schemas.py`

```python
class WatchlistName(BaseModel):
    name: str   # 1-50 chars, ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$

class StockSymbols(BaseModel):
    symbols: list[str]   # max 1000, each 1-10 chars, alphanumeric + dot + hyphen

class StockscreenJSONEncoder(json.JSONEncoder):
    # Handles: pd.Timestamp → ISO, pd.Period → str,
    #          date/datetime → ISO, NaN/Inf → None, numpy types → str
```

---

## 6. Dependency Injection — `create_services()`

All service singletons are instantiated once at module load via `create_services()` in `server.py`.
In tests, this function is patched to inject mocks.

```python
def create_services() -> tuple[ScreenerService, WatchlistService, NewsService, SymbolService, PalmaresService]:
    yahoo       = YahooProvider()
    boursorama  = BoursoramaProvider(cache_dir=DEFAULT_DATA_PATH, cache_ttl_seconds=86400)
    euronext    = EuronextProvider(cache_dir=DEFAULT_DATA_PATH, cache_ttl_seconds=EURONEXT_CACHE_TTL_SECONDS)
    facade      = MarketDataFacade(yahoo, boursorama, euronext)
    store       = ScreenerDataStore(base_path=DEFAULT_DATA_PATH)
    news        = NewsService(provider=facade)
    symbol_svc  = SymbolService(fetchers=build_fetchers(SYMBOL_SOURCES), cache_dir=DEFAULT_DATA_PATH, ...)
    screener    = ScreenerService(provider=facade, store=store, news_service=news, symbol_service=symbol_svc)
    watchlist   = WatchlistService(store=store)
    palmares    = PalmaresService(scraper=BoursoramaPalmaresScaper(), store=PalmaresStore(...), ...)
    return screener, watchlist, news, symbol_svc, palmares
```

Module-level singletons: `_screener, _watchlist, _news, _symbol_svc, _palmares_svc`.

---

## 7. Configuration — `stockscreen/config.py`

All configuration is read from environment variables at import time.

| Env var | Default | Description |
|---|---|---|
| `STOCKSCREEN_DATA_PATH` | `stockscreen/data/` | Root data directory |
| `STOCKSCREEN_TRANSPORT` | `stdio` | MCP transport: `stdio`, `sse`, or `streamable-http` |
| `STOCKSCREEN_HOST` | `127.0.0.1` | Host for HTTP transports (sse, streamable-http) |
| `STOCKSCREEN_PORT` | `8000` | Port for HTTP transports |
| `STOCKSCREEN_SYMBOL_SOURCES` | `sp500,nasdaq100,cac40,sbf120,dax,ftse100,aex` | Active index fetchers |
| `STOCKSCREEN_REFRESH_ON_STARTUP` | `true` | Seed caches at startup |
| `STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS` | `24` | Symbol cache TTL |
| `STOCKSCREEN_EURONEXT_CACHE_TTL` | `604800` (7 days) | ISIN↔ticker cache TTL |
| `STOCKSCREEN_PALMARES_CACHE_TTL` | `86400` (24 h) | Palmares snapshot TTL |

Log file: `stockscreen/stockscreen_v1.log` (also streams to stderr).
Logger name: `"stockscreen-server-v1"` — shared by all modules.

---

## 8. Async Patterns

All three external providers use **synchronous HTTP clients** (`requests`, `yfinance`) run inside `asyncio.get_event_loop().run_in_executor(None, lambda: ...)` to avoid blocking the event loop.

Services compose provider calls with `asyncio.gather` where parallelism is useful (e.g. Yahoo + Boursorama in `MarketDataFacade.get_quote`).

`BoursoramaPalmaresScaper.fetch_all` scrapes pages sequentially to avoid hammering the server.

`SymbolService.start_background_refresh` is an infinite coroutine launched as an `asyncio.create_task` in `_startup()`.

---

## 9. Data Persistence Layout

```
stockscreen/data/                    # DEFAULT_DATA_PATH
├── {category}.json                  # Symbol cache (sp500.json, cac40.json, ...)
│     {timestamp, symbols: [{symbol, name, market_cap, instrument_type}]}
├── euronext_{ISIN_or_ticker}.json   # ISIN↔ticker cache
│     {timestamp, data: {isin, symbol, name, mic, yahoo_ticker, cached_at}}
├── watchlists/
│   └── {name}.json                  # ["AAPL", "TTE.PA", ...]
├── screening_results/
│   └── {name}.json                  # Full run_stock_screen result dict
├── market_data/
│   └── default_symbols.json         # Legacy DefaultSymbols cache (unused)
└── palmares/
    └── palmares_dividendes.json     # PalmaresSnapshot
          {fetched_at, page_count, total_entries, entries: [PalmaresEntry]}
```

---

## 10. MCP Tools Summary

All tools defined in `server.py`. Each is an `async def` decorated with `@mcp.tool()`.

| Tool | Service method called | Saves to disk? |
|---|---|---|
| `run_stock_screen` | `_screener.run(...)` | Optional: `store.save_screening_result(save_result, ...)` |
| `get_stock_news` | `_news.get_news_data(...)` | No |
| `manage_watchlist` | `_watchlist.dispatch(...)` | Yes (watchlists/) |
| `get_screening_result` | `_screener.store.load_screening_result(...)` | No |
| `refresh_symbols` | `_symbol_svc.refresh(...)` | Yes (category .json) |
| `get_palmares` | `_palmares_svc.get/refresh(...)` | Yes (palmares/ on refresh) |

---

## 11. Testing

Test runner: `pytest` with `asyncio_mode = "auto"` (async tests run without explicit `@pytest.mark.asyncio`).

All provider calls are **mocked** — no real network calls in the test suite.

Key files:

| File | What it covers |
|---|---|
| `tests/test_server.py` | FastMCP tool routing, error paths |
| `tests/test_screener_service.py` | ScreenerService — all 5 screen types |
| `tests/test_market_data_facade.py` | Facade merge logic, ISIN detection, fallbacks |
| `tests/test_boursorama_provider.py` | Scraping, cache read/write, stale fallback |
| `tests/test_euronext_provider.py` | Bidirectional resolution, cache, TTL |
| `tests/test_boursorama_palmares.py` | HTML parsing, page detection, row errors |
| `tests/test_palmares_store.py` | PalmaresStore save/load |
| `tests/test_palmares_service.py` | Cache freshness, sort, filter, limit |
| `tests/test_symbol_service.py` | get(), refresh(), background refresh |
| `tests/test_watchlist_service.py` | CRUD dispatch, validation |
| `tests/test_news_service.py` | get_news_data(), screen_news() |
| `tests/test_yahoo_provider.py` | Retry decorator, run_in_executor wrapping |
| `tests/conftest.py` | Shared fixtures (tmp paths, mock providers) |

Run:
```bash
uv run pytest                                         # all tests
uv run pytest tests/test_screener_service.py          # single module
uv run pytest --cov=stockscreen --cov-report=term-missing  # with coverage
```
