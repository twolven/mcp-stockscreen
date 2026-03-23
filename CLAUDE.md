# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockScreen is a single-file Python MCP (Model Context Protocol) server that provides stock screening tools to Claude Desktop. It fetches real-time market data via Yahoo Finance (`yfinance`) and exposes four MCP tools: `run_stock_screen`, `get_stock_news`, `manage_watchlist`, and `get_screening_result`.

## Running

```bash
pip install -r requirements.txt
python stockscreen.py
```

The server communicates via stdio-based MCP protocol — it's designed to be launched by Claude Desktop, not run interactively. There are no tests.

## Architecture

Everything lives in `stockscreen.py` (~1580 lines). Key components:

- **`DefaultSymbols`** — Manages cached symbol lists from Yahoo Finance indices (S&P 500, Dow, Nasdaq), categorized by market cap (mega/large/mid/small/micro cap, ETFs). Cache expires after 24 hours.
- **`ScreenerDataStore`** — JSON file-based persistence for watchlists and screening results under `data/`.
- **MCP Server (`app`)** — Registers tools via `@app.list_tools()` and `@app.call_tool()` decorators. Runs on `mcp.server.stdio`.
- **Screening functions** — `run_technical_screen()`, `run_fundamental_screen()`, `run_options_screen()`, `run_news_screen()`, `run_custom_screen()` each implement a distinct screening strategy.

## Data Flow

Claude Desktop → MCP stdio → tool routing → screening functions → yfinance API → pandas processing → JSON storage/response

## Key Details

- Data stored in `data/` relative to script directory (screening_results/, watchlists/, market_data/), auto-created on first run
- `STOCKSCREEN_DATA_PATH` env var overrides the default data path
- `retry_on_error()` decorator provides exponential backoff (3 attempts) for Yahoo Finance calls
- `StockscreenJSONEncoder` handles pandas Timestamp/NaT and numpy types for JSON serialization
- Symbol validation caps at 1000 symbols per request
- Watchlist names: 1-50 chars, alphanumeric plus `_` and `-`
