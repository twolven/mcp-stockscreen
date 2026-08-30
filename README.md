# StockScreen MCP

A typed FastMCP server for Yahoo Finance-backed technical, fundamental, options, news, custom, and watchlist screening. It supports local stdio and containerized Streamable HTTP transports.

## Tools

- `run_stock_screen(screen_type, criteria, watchlist=None, save_result=None)`
- `get_stock_news(symbol, days_back=30)`
- `manage_watchlist(action, name, symbols=None)`
- `get_screening_result(name)`

Legacy inputs and the five original screen categories are retained. Unknown criteria are rejected rather than silently ignored. Technical screens support the original price, volume, SMA-50/SMA-200, RSI, and ATR criteria. Fundamental screens include minimum/maximum market cap, P/E, dividend, revenue growth, profit margin, debt/equity, price/book, AUM, expense ratio, and volume. Options screens include IV, option volume, put/call ratio, spread, expiration-day, and earnings-day bounds. ETF, news, and nested custom criteria retain their original names. Explicit symbols or a watchlist are preferred. With neither, the server uses yfinance's documented `screen` query with `size=250`, US-region filters, category-aware EquityQuery/ETFQuery predicates, and assets/market-cap descending order. Universe metadata includes source, category, requested/returned size, sort, and as-of time. Missing requested metrics reject a symbol instead of becoming zero, and every rejected symbol retains reasons. Persistence names are restricted and normalized JSON writes are atomic under `~/.stockscreen`; no migration runs during import.

```powershell
uv sync --locked
uv run python stockscreen.py
```

The server uses stdio and emits no logs or migration messages to stdout. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, or structurally changed. Results are not investment advice or guaranteed real-time data.

## Docker / Streamable HTTP

The container runs as an unprivileged user, installs the locked production dependencies, and serves MCP at `http://127.0.0.1:8000/mcp`. Start it with:

```powershell
docker compose up --build -d
Invoke-RestMethod http://127.0.0.1:8000/health
```

Connect a Streamable HTTP-capable MCP client to `http://127.0.0.1:8000/mcp`. The named `stockscreen-data` volume persists watchlists and saved results across container replacement. To avoid a port collision when running multiple servers, set `MCP_HOST_PORT` before starting Compose, for example `$env:MCP_HOST_PORT=8002`. Stop the container with `docker compose down`; add `--volumes` only when you intentionally want to delete the persisted data.

The Compose mapping intentionally binds to localhost. The endpoint has no authentication or TLS and must not be exposed to an untrusted network without a properly configured reverse proxy and access control.

Binding to loopback alone does not make the endpoint private: a browser can still reach it through DNS rebinding, so the server validates `Host` and `Origin` headers before a request reaches an MCP session. Requests carrying a foreign `Host` are answered with `421 Misdirected Request` and those carrying a foreign `Origin` with `403 Forbidden`, while same-origin loopback traffic and non-browser clients that send no `Origin` are unaffected.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | Interface the HTTP server binds. |
| `MCP_PORT` | `8000` | Port inside the container. |
| `MCP_PATH` | `/mcp` | Streamable HTTP endpoint path. |
| `MCP_HOST_PORT` | `8000` | Host port Compose publishes on `127.0.0.1`. |
| `MCP_HOST_ORIGIN_PROTECTION` | `true` | `true`, `auto`, or `false`. Disable only behind a proxy that performs the same validation. |
| `MCP_ALLOWED_HOSTS` | unset | Comma-separated extra hostnames permitted in `Host`. |
| `MCP_ALLOWED_ORIGINS` | unset | Comma-separated extra browser origins permitted in `Origin`. |

Put the reverse-proxy hostname in `MCP_ALLOWED_HOSTS` when fronting the container, otherwise the guard rejects the proxied `Host`. Running `uv run python stockscreen.py` remains the stdio-compatible default outside Docker.

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. CI also builds the container and performs health plus MCP tool-discovery checks over Streamable HTTP. Domain/provider/persistence branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into live shape smoke tests.
