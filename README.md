# StockScreen MCP

A typed FastMCP stdio server for Yahoo Finance-backed technical, fundamental, options, news, custom, and watchlist screening.

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

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. Domain/provider/persistence branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into live shape smoke tests.
