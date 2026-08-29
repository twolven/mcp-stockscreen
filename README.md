# StockScreen MCP

A typed FastMCP stdio server for Yahoo Finance-backed technical, fundamental, options, news, custom, and watchlist screening.

## Tools

- `run_stock_screen(screen_type, criteria, watchlist=None, save_result=None)`
- `get_stock_news(symbol, days_back=30)`
- `manage_watchlist(action, name, symbols=None)`
- `get_screening_result(name)`

Legacy inputs and the five original screen categories are retained. Explicit symbols or a watchlist are preferred. With neither, the server uses yfinance's documented `screen`/`EquityQuery` API and records the universe and as-of time; Yahoo limits custom-query results to 250. Missing requested metrics reject a symbol instead of becoming zero, and every rejected symbol retains reasons. Persistence names are restricted and JSON writes are atomic under `~/.stockscreen`; no migration runs during import.

```powershell
uv sync --locked
uv run python stockscreen.py
```

The server uses stdio and emits no logs or migration messages to stdout. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, or structurally changed. Results are not investment advice or guaranteed real-time data.

Run validation with `uv run ruff check .`, `uv run pytest`, and `uv build`.
