# StockScreen MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server for stock screening. Expose six tools to any MCP-compatible client (Claude Code, Claude Desktop…) to screen stocks on technical, fundamental, options, and news criteria, manage watchlists, retrieve dividend rankings, and persist results.

**Data sources**
- **Yahoo Finance** (`YahooProvider`) — primary source for price, volume, RSI, P/E, options chains, and news. Covers US and international equities and ETFs.
- **Boursorama** (`BoursoramaProvider`) — supplementary provider for Euronext Paris data (cours, dividende, rendement, consensus analystes, historique CA/RN). Integrated transparently via `MarketDataFacade`.
- **Euronext** (`EuronextProvider`) — bidirectional ISIN ↔ ticker resolution. Converts ISIN identifiers to Yahoo-compatible tickers (e.g. `FR0000131104` → `TTE.PA`).
- **Boursorama Palmares** (`BoursoramaPalmaresScaper`) — scrapes the full multi-page Boursorama dividend ranking table (palmarès dividendes Euronext Paris).

---

## Features

| Category | Criteria |
|---|---|
| **Technical** | Price, volume, RSI(14), SMA 20/50/200, ATR%, trend changes |
| **Fundamental** | Market cap, P/E, dividend yield, revenue growth, ETF metrics (AUM, expense ratio) |
| **Options** | IV, option volume, put/call ratio, bid-ask spread, days-to-earnings |
| **News** | Keyword matching, management changes, date range |
| **Custom** | Combine any of the above — short-circuits on first rejection |
| **Dividend palmares** | Boursorama top-yield ranking with multi-year dividend history, filterable by yield and name |

---

## Installation

**Only prerequisite**: [uv](https://docs.astral.sh/uv/)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No Python installation required — uv manages it automatically.

---

## Connecting to Claude Code

Three ways to wire the server, from simplest to most involved.

### Option 1 — `uvx` from GitHub (no clone needed)

```bash
claude mcp add stockscreen -- uvx --from git+https://github.com/YOUR_USER/mcp-stockscreen stockscreen
```

Or via `.mcp.json` at the root of any project:

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/cyberbobjr/mcp-stockscreen", "stockscreen"]
    }
  }
}
```

`uvx` fetches the package, creates an isolated environment, and launches the server — nothing to clone or install first.

### Option 2 — `uvx` from PyPI (if published)

```bash
claude mcp add stockscreen -- uvx mcp-stockscreen
```

Or via `.mcp.json`:

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uvx",
      "args": ["mcp-stockscreen"]
    }
  }
}
```

### Option 3 — Local clone

```bash
git clone https://github.com/cyberbobjr/mcp-stockscreen.git
cd mcp-stockscreen
uv sync
```

```bash
claude mcp add stockscreen -- uv --directory /absolute/path/to/mcp-stockscreen run stockscreen
```

Or via `.mcp.json`:

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-stockscreen", "run", "stockscreen"]
    }
  }
}
```

Verify the server is registered:

```bash
claude mcp list
```

> The server reads the `STOCKSCREEN_DATA_PATH` environment variable to override the default data directory. Add it to the `env` block of any config above:
> ```json
> "env": { "STOCKSCREEN_DATA_PATH": "/path/to/data" }
> ```

---

## Connecting to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows).

**From GitHub (no clone needed):**

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/YOUR_USER/mcp-stockscreen", "stockscreen"]
    }
  }
}
```

**From PyPI (if published):**

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uvx",
      "args": ["mcp-stockscreen"]
    }
  }
}
```

**From a local clone:**

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-stockscreen", "run", "stockscreen"]
    }
  }
}
```

Restart Claude Desktop — the tools will appear in the tools panel.

---

## Available Tools

### `run_stock_screen`

Screen a list of stocks and return those that match all criteria.

**Technical screen** — price, volume, momentum indicators:
```json
{
  "screen_type": "technical",
  "criteria": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "min_price": 10.0,
    "max_price": 500.0,
    "min_volume": 500000,
    "min_rsi": 30,
    "max_rsi": 70,
    "above_sma_200": true,
    "above_sma_50": false,
    "max_atr_pct": 5.0,
    "category": "large_cap"
  }
}
```

**Fundamental screen** — valuation and growth:
```json
{
  "screen_type": "fundamental",
  "criteria": {
    "min_market_cap": 10000000000,
    "max_pe": 30,
    "min_dividend": 2.0,
    "min_revenue_growth": 0.05,
    "category": "mega_cap"
  }
}
```

**Options screen** — volatility and flow:
```json
{
  "screen_type": "options",
  "criteria": {
    "min_iv": 20,
    "max_iv": 80,
    "min_option_volume": 5000,
    "min_put_call_ratio": 0.5,
    "max_spread": 10.0,
    "min_days_to_earnings": 7,
    "max_days_to_earnings": 30
  }
}
```

**News screen** — keyword and event matching:
```json
{
  "screen_type": "news",
  "criteria": {
    "keywords": ["acquisition", "merger"],
    "exclude_keywords": ["lawsuit"],
    "require_all_keywords": false,
    "min_days": 0,
    "max_days": 14,
    "management_changes": false
  }
}
```

**Custom screen** — combine any criteria (short-circuits on first rejection):
```json
{
  "screen_type": "custom",
  "criteria": {
    "symbols": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "technical": { "above_sma_200": true, "max_rsi": 65 },
    "fundamental": { "min_market_cap": 100000000000, "max_pe": 30 }
  }
}
```

**Additional parameters:**
- `watchlist` (string) — use a saved watchlist as the symbol source
- `save_result` (string) — persist the result under this name for later retrieval

**Response:**
```json
{
  "screen_type": "technical",
  "criteria": {},
  "matches": 2,
  "results": [
    {
      "symbol": "AAPL",
      "price": 175.0,
      "volume": 55000000,
      "rsi": 52.3,
      "sma_20": 172.0,
      "sma_50": 168.0,
      "sma_200": 155.0,
      "atr": 3.2,
      "atr_pct": 1.83,
      "price_changes": { "1d": 0.5, "5d": 2.1, "20d": 4.8 },
      "ma_distances": { "pct_from_20sma": 1.7, "pct_from_50sma": 4.2, "pct_from_200sma": 12.9 }
    }
  ],
  "rejected": [
    { "symbol": "XYZ", "rejection_reasons": ["RSI 78.2 > max_rsi 70"] }
  ],
  "timestamp": "2026-03-24T10:00:00"
}
```

---

### `get_stock_news`

Get recent news and company updates for a single ticker.

```
symbol: "AAPL"         # US ticker, or "TTE.PA" for Euronext Paris, "DBK.DE" for Xetra, etc.
days_back: 30          # default 30, max ~90
```

Returns:
- `recent_news` — all articles within the date window (title, publisher, published, summary, url)
- `key_events` — subset matching regulatory/legal keywords (SEC, lawsuit, investigation…)
- `management_changes` — subset matching executive keywords (CEO, CFO, chairman…)
- `current_management` — company officers from Yahoo
- `company_info` — sector, industry, country, website, employees, marketCap

---

### `manage_watchlist`

Create, update, delete, or retrieve a named watchlist.

| action | description | requires `symbols` |
|---|---|---|
| `create` | create new watchlist | yes |
| `update` | replace symbols | yes |
| `get` | return symbol list | no |
| `delete` | remove watchlist | no |

```
action: "create"
name: "cac-picks"                        # 1–50 chars, alphanumeric + _ and -
symbols: ["TTE.PA", "AIR.PA", "MC.PA"]
```

Typical workflow — create a watchlist, then screen it:
```json
// 1. Save the list once
{ "action": "create", "name": "cac-picks", "symbols": ["TTE.PA", "AIR.PA", "MC.PA"] }

// 2. Screen it any time
{ "screen_type": "fundamental", "criteria": { "min_dividend": 3.0 }, "watchlist": "cac-picks" }
```

---

### `get_screening_result`

Retrieve a result previously saved via `run_stock_screen`'s `save_result` parameter.

```
name: "cac40-div-2026-03"
```

Returns the full original screening result dict (screen_type, criteria, matches, results, rejected, timestamp), or `{"error": "..."}` if not found.

---

### `get_palmares`

Get the Boursorama dividend palmares — French equities ranked by dividend yield, scraped from [Boursorama palmarès dividendes](https://www.boursorama.com/bourse/actions/palmares/dividendes/).

Results are sorted by best rendement descending. The snapshot is cached for 24 h.

```
min_rendement: 3.0       # minimum yield in % (optional)
max_rendement: 10.0      # maximum yield in % (optional)
nom_contains: "total"    # case-insensitive name filter (optional)
limit: 50                # max entries returned, capped at 500 (default 50)
force_refresh: false     # force a fresh scrape (default false)
```

**Response:**
```json
{
  "fetched_at": "2026-03-24T10:00:00",
  "total_entries": 312,
  "returned": 20,
  "entries": [
    {
      "code_bourso": "1rTTE",
      "nom": "TotalEnergies SE",
      "cours": 62.5,
      "isin": null,
      "dividendes": [
        { "annee": "2025", "dividende": 3.22, "rendement": 5.15 },
        { "annee": "2026", "dividende": 3.40, "rendement": 5.44 }
      ]
    }
  ]
}
```

**Typical workflow — palmares → technical cross-check:**
```json
// 1. Get top 30 high-yield French stocks
get_palmares(min_rendement=5.0, limit=30)

// 2. Cross-check technicals on the results
run_stock_screen(
  screen_type="technical",
  criteria={"symbols": ["1rTTE", "1rMC", ...], "above_sma_200": true}
)
```

> **Note:** `code_bourso` values (e.g. `1rTTE`) work as screening symbols when Boursorama data is available through the facade. For Yahoo-based screening, resolve to the Yahoo ticker first (e.g. `TTE.PA`).

---

### `refresh_symbols`

Force a refresh of the cached symbol lists for one or all index categories. Useful after a recent constituent change or when a screening run returned fewer symbols than expected.

```
category: "cac40"   # or omit to refresh all active sources at once
```

Supported categories: `sp500`, `nasdaq100`, `cac40`, `sbf120`, `dax`, `ftse100`, `aex`.

Returns `{ "cac40": 40 }` (category → number of symbols fetched), or `{ "cac40": { "error": "..." } }` on failure.

---

## Symbol categories

Use the `category` key in criteria to screen against a built-in index universe. Symbol lists are fetched from Wikipedia and cached locally (TTL configurable, default 24 h).

**Market-cap buckets** (US indices — S&P 500 + Nasdaq 100 combined):

| value | market cap |
|---|---|
| `mega_cap` | > $200 B |
| `large_cap` | $10 B – $200 B |
| `mid_cap` | $2 B – $10 B |
| `small_cap` | $300 M – $2 B |
| `micro_cap` | < $300 M |
| `etf` | ETFs |

**Index categories** (full constituent list, tickers with exchange suffix):

| value | index | exchange suffix |
|---|---|---|
| `sp500` | S&P 500 | *(none)* |
| `nasdaq100` | Nasdaq 100 | *(none)* |
| `cac40` | CAC 40 | `.PA` |
| `sbf120` | SBF 120 | `.PA` |
| `dax` | DAX | `.DE` |
| `ftse100` | FTSE 100 | `.L` |
| `aex` | AEX | `.AS` |

Example — screen all CAC 40 stocks for dividend yield ≥ 3%:
```json
{
  "screen_type": "fundamental",
  "criteria": { "category": "cac40", "min_dividend": 3.0 }
}
```

---

## Data sources

### Yahoo Finance (primary — all screening tools)

All screening tools use `YahooProvider` wrapped by `MarketDataFacade`, which adds:
- True async via `asyncio.run_in_executor`
- Exponential-backoff retry (3 attempts)
- Automatic Boursorama enrichment for dividend/yield fields

**Known limitation**: `dividendYield` from Yahoo is inconsistent for non-US stocks. The facade overrides it with Boursorama data when available.

### Boursorama (dividends + cours — Euronext Paris)

`BoursoramaProvider` scrapes Boursorama for reliable Euronext dividend data. It is called automatically by `MarketDataFacade` for every quote — Boursorama is tried first for `dividende`, `rendement`, `last_dividend_date`, `consensus`, and `performance`; Yahoo is used as fallback.

Accepts either ISIN (e.g. `FR0000131104`) or Boursorama code (e.g. `1rTTE`) as input. Cache: one JSON file per ticker in `data/`, TTL 24 h.

**`BoursoramaQuote` fields:**

| Field | Type | Description |
|---|---|---|
| `isin` | str | ISIN or code used for the lookup |
| `code_bourso` | str | Boursorama internal code (e.g. `1rTTE`) |
| `nom` | str | Company name |
| `lien` | str | URL of the Boursorama page |
| `cours` | float \| None | Last price |
| `dividende` | float \| None | Annual dividend (EUR) |
| `rendement` | float \| None | Yield `dividende / cours × 100` (%) |
| `last_dividend_date` | str \| None | Last detachment date (ISO) |
| `consensus` | str \| None | Analyst consensus label |
| `performance` | list[dict] | `[{annee, ca, rn, marge}]` per year |
| `cached_at` | str | ISO timestamp of last fetch |

### Euronext (ISIN ↔ ticker resolution)

`EuronextProvider` resolves identifiers bidirectionally:
- `resolve_ticker(isin)` → `EuronextRecord` (with `yahoo_ticker` like `TTE.PA`)
- `resolve_isin(ticker)` → `EuronextRecord` (with `isin` like `FR0000131104`)

Cache is shared between both directions: resolving a ticker also caches the result under the ISIN key. TTL: 7 days (configurable via `STOCKSCREEN_EURONEXT_CACHE_TTL`).

Supported exchange suffixes: `.PA` (Euronext Paris), `.DE` (Xetra), `.L` (LSE), `.AS` (Amsterdam), `.MI` (Milan), `.MC` (Madrid), `.BR` (Brussels), `.LS` (Lisbon), `.HE` (Helsinki), `.ST` (Stockholm), `.OL` (Oslo).

### Boursorama Palmares (dividend ranking)

`BoursoramaPalmaresScaper` scrapes the full multi-page Boursorama dividend ranking table. Data includes up to 3 years of dividend history per stock. Cached as a single JSON snapshot (TTL: 24 h, configurable via `STOCKSCREEN_PALMARES_CACHE_TTL`).

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `STOCKSCREEN_DATA_PATH` | `stockscreen/data/` | Root directory for all stored data |
| `STOCKSCREEN_SYMBOL_SOURCES` | `sp500,nasdaq100,cac40,sbf120,dax,ftse100,aex` | Comma-separated list of active index fetchers |
| `STOCKSCREEN_REFRESH_ON_STARTUP` | `true` | Fetch missing/stale symbol caches at startup |
| `STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS` | `24` | Cache TTL for symbol lists (hours) |
| `STOCKSCREEN_EURONEXT_CACHE_TTL` | `604800` (7 days) | ISIN/ticker resolution cache TTL (seconds) |
| `STOCKSCREEN_PALMARES_CACHE_TTL` | `86400` (24 h) | Palmares snapshot cache TTL (seconds) |

---

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
│   ├── facade.py                      # MarketDataFacade — single entry point (Yahoo+Boursorama+Euronext)
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
│   └── palmares_service.py            # PalmaresService — cache, filter, sort palmares
└── store/
    ├── data_store.py                  # ScreenerDataStore + DefaultSymbols (JSON persistence)
    └── palmares_store.py              # PalmaresStore — palmares snapshot persistence
```

### Data flow

```
Claude Desktop
  → FastMCP stdio
    → @mcp.tool() (server.py)
      → Service (screener / news / watchlist / palmares)
        → MarketDataFacade
          ├── YahooProvider  → yfinance → pandas → JSON
          ├── BoursoramaProvider → HTTP scrape → merge (Boursorama-first for dividends)
          └── EuronextProvider → HTTP → ISIN↔ticker resolution

        → BoursoramaPalmaresScaper → HTTP multi-page scrape → PalmaresStore (JSON)
        → SymbolService → Wikipedia fetchers → JSON cache
        → ScreenerDataStore → JSON files (watchlists, screening results)
```

Data is stored under `data/` (overridable with `STOCKSCREEN_DATA_PATH`).

---

## Development

```bash
uv sync --group dev
uv run pytest          # ~290 tests, no network calls (all providers mocked)
uv run pytest tests/test_screener_service.py   # specific module
```

---

## Limitations

- **Yahoo Finance**: potential delays and rate limits; dividend data unreliable for non-US stocks (mitigated by Boursorama enrichment)
- **Boursorama**: scraping-based — may break if Boursorama changes its HTML structure; reliable data for Euronext Paris equities only
- **Euronext provider**: coverage limited to equities listed on the 11 supported MIC exchanges; no US stocks
- **Palmares**: scraping-based — covers only Euronext Paris equities; does not include sector or compartment data (not present in the palmares table)
- Options data depends on market hours and symbol coverage
- Symbol index lists (CAC 40, etc.) are fetched from Wikipedia — may lag a few days after constituent changes

---

## License

MIT — see [LICENSE](LICENSE).
