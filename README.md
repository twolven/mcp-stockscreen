# StockScreen MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server for stock screening. Expose four tools to any MCP-compatible client (Claude Code, Claude Desktop…) to screen stocks on technical, fundamental, options, and news criteria, manage watchlists, and persist results.

**Data sources**
- **Yahoo Finance** (`YahooProvider`) — primary source used by all screening tools. Covers US and international equities, ETFs, options chains, and news.
- **Boursorama** (`BoursoramaProvider`) — supplementary provider for Euronext Paris data (cours, dividende, rendement, consensus analystes, historique CA/RN). Useful when Yahoo's dividend data is unreliable for French stocks. **Not wired into the screening tools by default** — called directly via Python API (see [Data sources](#data-sources-detail) below).

---

## Features

| Category | Criteria |
|---|---|
| **Technical** | Price, volume, RSI(14), SMA 20/50/200, ATR%, trend changes |
| **Fundamental** | Market cap, P/E, dividend yield, revenue growth, ETF metrics (AUM, expense ratio) |
| **Options** | IV, option volume, put/call ratio, bid-ask spread, days-to-earnings |
| **News** | Keyword matching, management changes, date range |
| **Custom** | Combine any of the above — short-circuits on first rejection |

---

## Installation

**Requirements**: Python 3.10+

```bash
git clone https://github.com/twolven/mcp-stockscreen.git
cd mcp-stockscreen
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e .
```

---

## Connecting to Claude Code

### Option 1 — `claude mcp add` (recommended)

```bash
# From the project directory, with venv activated:
claude mcp add stockscreen -- /absolute/path/to/venv/bin/python -m stockscreen.server
```

Or if the package is installed system-wide:

```bash
claude mcp add stockscreen -- stockscreen
```

Verify the server is registered:

```bash
claude mcp list
```

### Option 2 — `.mcp.json` in the project

Create (or edit) `.mcp.json` at the root of any project where you want the tools available:

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "stockscreen.server"]
    }
  }
}
```

> The server reads the `STOCKSCREEN_DATA_PATH` environment variable to override the default data directory. You can set it in the `env` block:
> ```json
> "env": { "STOCKSCREEN_DATA_PATH": "/path/to/data" }
> ```

---

## Connecting to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "stockscreen": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "stockscreen.server"]
    }
  }
}
```

Restart Claude Desktop — the four tools will appear in the tools panel.

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
symbol: "AAPL"
days_back: 30
```

Returns `recent_news`, `key_events`, `management_changes`, `current_management`, `company_info`.

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
name: "tech-picks"
symbols: ["AAPL", "MSFT", "GOOGL"]
```

Watchlist names: 1–50 characters, alphanumeric plus `_` and `-`.

---

### `get_screening_result`

Retrieve a result previously saved via `run_stock_screen`'s `save_result` parameter.

```
name: "my-screen-run"
```

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

You can force-refresh symbol lists with the `refresh_symbols` tool (see below).

---

## Available Tools (full list)

| Tool | Description |
|---|---|
| `run_stock_screen` | Screen stocks by technical / fundamental / options / news / custom criteria |
| `get_stock_news` | Get recent news for a ticker |
| `manage_watchlist` | Create / update / delete / get a named watchlist |
| `get_screening_result` | Retrieve a previously saved screening result |
| `refresh_symbols` | Force-refresh the symbol cache for one or all index categories |

### `refresh_symbols`

```
category: "cac40"   # or omit to refresh all sources
```

Returns `{ "cac40": 40 }` (category → number of symbols fetched), or `{ "cac40": { "error": "..." } }` on failure.

---

## Data sources detail

### Yahoo Finance (primary — all screening tools)

All four screening tools (`run_stock_screen`, `get_stock_news`, …) exclusively use `YahooProvider`, which wraps `yfinance` with:
- True async via `asyncio.run_in_executor`
- Exponential-backoff retry (3 attempts)

**Known limitation**: `dividendYield` from Yahoo is inconsistent for non-US stocks (sometimes decimal, sometimes already a %). The screener normalises it via `dividendRate / price` when available, then falls back to a format-detection heuristic.

### Boursorama (supplementary — Python API only)

`BoursoramaProvider` scrapes Boursorama for more reliable Euronext data. It is **not called automatically** by the screening tools — you must instantiate it directly in Python:

```python
from stockscreen.providers.boursorama import BoursoramaProvider

provider = BoursoramaProvider(
    cache_dir="/path/to/cache",
    cache_ttl_seconds=86400,    # 24 h
    exchange_filter="Euronext", # None = all exchanges
)

quote = await provider.get_quote("FR0000131104")  # TotalEnergies ISIN
print(quote.dividende)      # annual dividend in EUR
print(quote.rendement)      # yield in %
print(quote.consensus)      # analyst consensus label
print(quote.performance)    # [{annee, ca, rn, marge}, ...]
```

**`BoursoramaQuote` fields:**

| Field | Type | Description |
|---|---|---|
| `isin` | str | ISIN used for the lookup |
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

**Cache**: one JSON file per ISIN in `cache_dir`. On network failure, stale cache is served as fallback. Call `provider.invalidate_cache(isin)` to force a fresh fetch.

**Exchange filter**: set `exchange_filter=None` to accept results from all exchanges (NYSE, XETRA, LSE, …). Coverage outside Euronext is limited to ETFs and derivatives.

**Switching between providers**: there is no automatic fallback between Yahoo and Boursorama. If you want to enrich Yahoo screening results with Boursorama dividend data, call both independently and merge:

```python
# Yahoo for screening
result = await screener.run("fundamental", {"category": "cac40", "min_dividend": 2.0})

# Boursorama for accurate dividend data on matches
bourso = BoursoramaProvider(cache_dir=cache_dir)
for stock in result["results"]:
    quote = await bourso.get_quote(stock["isin"])   # requires ISIN mapping
    stock["dividende_bourso"] = quote.rendement
```

> A composite provider with automatic Yahoo → Boursorama fallback is not yet implemented.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `STOCKSCREEN_DATA_PATH` | `stockscreen/data/` | Root directory for all stored data |
| `STOCKSCREEN_SYMBOL_SOURCES` | `sp500,nasdaq100,cac40,sbf120,dax,ftse100,aex` | Comma-separated list of active index fetchers |
| `STOCKSCREEN_REFRESH_ON_STARTUP` | `true` | Fetch missing/stale symbol caches at startup |
| `STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS` | `24` | Cache TTL for symbol lists (hours) |

---

## Architecture

```
stockscreen/
├── server.py                      # FastMCP tools → services
├── config.py                      # Paths, logging, env overrides
├── exceptions.py                  # StockscreenError, ValidationError, APIError
├── providers/
│   ├── yahoo.py                   # YahooProvider — yfinance wrapper (async)
│   ├── boursorama.py              # BoursoramaProvider — Boursorama scraper (async)
│   └── symbol_fetchers/
│       ├── base.py                # BaseSymbolFetcher ABC + SymbolRecord
│       ├── wikipedia.py           # SP500, Nasdaq100, CAC40, SBF120, DAX, FTSE100, AEX
│       └── registry.py            # build_fetchers(["cac40", "sp500", ...])
├── models/schemas.py              # Pydantic v2 validation + JSON encoder
├── services/
│   ├── screener.py                # ScreenerService (technical/fundamental/options/news/custom)
│   ├── news.py                    # NewsService
│   ├── watchlist.py               # WatchlistService
│   └── symbol_service.py          # SymbolService — fetch/cache/refresh index symbol lists
└── store/data_store.py            # ScreenerDataStore + DefaultSymbols (JSON persistence)
```

Data is stored under `data/` (overridable with `STOCKSCREEN_DATA_PATH`).

---

## Development

```bash
source venv/bin/activate
pytest          # 290 tests, no network calls
```

---

## Limitations

- **Yahoo Finance**: potential delays and rate limits; dividend data unreliable for non-US stocks
- **Boursorama**: scraping-based — may break if Boursorama changes its HTML structure; Euronext-only for reliable data; requires ISIN (not ticker) as input
- Options data depends on market hours and symbol coverage
- Symbol index lists (CAC 40, etc.) are fetched from Wikipedia — may lag a few days after constituent changes

---

## License

MIT — see [LICENSE](LICENSE).
