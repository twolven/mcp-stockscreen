# StockScreen MCP Server

A Model Context Protocol (MCP) server providing comprehensive stock screening capabilities through Yahoo Finance. Enables LLMs to screen stocks based on technical, fundamental, and options criteria, with support for watchlist management and result storage.

## Features

### Stock Screening
- Technical Analysis Screening
  - Price and volume filters
  - Moving averages (20, 50, 200 SMA)
  - RSI indicators
  - Average True Range (ATR)
  - Trend analysis (1d, 5d, 20d changes)
  - MA distance calculations

- Fundamental Screening
  - Market capitalization filters
  - P/E ratio analysis
  - Dividend yield criteria
  - Revenue growth metrics
  - ETF-specific metrics (AUM, expense ratio)

- Options Screening
  - Implied Volatility (IV) filters
  - Options volume and open interest
  - Put/Call ratio analysis
  - Bid-ask spread evaluation
  - Earnings date proximity checks

### Data Management
- Watchlist Creation and Management
- Screening Result Storage
- Default Symbol Categories
  - Mega Cap (>$200B)
  - Large Cap ($10B-$200B)
  - Mid Cap ($2B-$10B)
  - Small Cap ($300M-$2B)
  - Micro Cap (<$300M)
  - ETFs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/twolven/mcp-stockscreen.git
cd mcp-stockscreen
```

## Usage

1. Add to your Claude configuration:
In your `claude-desktop-config.json`, add the following to the `mcpServers` section:

```json
{
    "mcpServers": {
        "stockscreen": {
            "command": "python",
            "args": ["path/to/stockscreen.py"]
        }
    }
}
```

Replace "path/to/stockscreen.py" with the full path to where you saved the stockscreen.py file.

## Available Tools

## Available Tools

1. `run_stock_screen`
   
### Technical Screen Criteria
```python
{
    "screen_type": "technical",
    "criteria": {
        "min_price": float,              # Minimum stock price
        "max_price": float,              # Maximum stock price
        "min_volume": int,               # Minimum average volume
        "above_sma_200": bool,           # Price above 200-day SMA
        "above_sma_50": bool,            # Price above 50-day SMA
        "min_rsi": float,                # Minimum RSI value
        "max_rsi": float,                # Maximum RSI value
        "max_atr_pct": float,            # Maximum ATR as percentage of price
        "category": str                  # Optional: market cap category filter
    },
    "watchlist": str,                    # Optional: name of watchlist to screen
    "save_result": str                   # Optional: name to save results
}
```

### Fundamental Screen Criteria
```python
{
    "screen_type": "fundamental",
    "criteria": {
        "min_market_cap": float,         # Minimum market capitalization
        "min_pe": float,                 # Minimum P/E ratio
        "max_pe": float,                 # Maximum P/E ratio
        "min_dividend": float,           # Minimum dividend yield (%)
        "min_revenue_growth": float,     # Minimum revenue growth rate
        "category": str,                 # Optional: market cap category filter
        
        # ETF-specific criteria
        "min_aum": float,                # Minimum assets under management
        "max_expense_ratio": float,      # Maximum expense ratio
        "min_volume": float              # Minimum trading volume
    },
    "watchlist": str,                    # Optional: name of watchlist to screen
    "save_result": str                   # Optional: name to save results
}
```

### Options Screen Criteria
```python
{
    "screen_type": "options",
    "criteria": {
        "min_iv": float,                 # Minimum implied volatility (%)
        "max_iv": float,                 # Maximum implied volatility (%)
        "min_option_volume": int,        # Minimum options volume
        "min_put_call_ratio": float,     # Minimum put/call ratio
        "max_spread": float,             # Maximum bid-ask spread (%)
        "min_days_to_earnings": int,     # Minimum days until earnings
        "max_days_to_earnings": int,     # Maximum days until earnings
        "category": str                  # Optional: market cap category filter
    },
    "watchlist": str,                    # Optional: name of watchlist to screen
    "save_result": str                   # Optional: name to save results
}
```

### News Screen Criteria
```python
{
    "screen_type": "news",
    "criteria": {
        "keywords": List[str],           # Keywords to search for in news
        "exclude_keywords": List[str],    # Keywords to exclude from results
        "min_days": int,                 # Minimum days back to search
        "max_days": int,                 # Maximum days back to search
        "management_changes": bool,       # Filter for management changes
        "require_all_keywords": bool,     # Require all keywords to match
        "category": str                  # Optional: market cap category filter
    },
    "watchlist": str,                    # Optional: name of watchlist to screen
    "save_result": str                   # Optional: name to save results
}

```

### Custom Screen Criteria
```python
{
    "screen_type": "custom",
    "criteria": {
        "category": str,                 # Optional: market cap category filter
        "technical": {
            # Any technical criteria from above
        },
        "fundamental": {
            # Any fundamental criteria from above
        },
        "options": {
            # Any options criteria from above
        }
    },
    "watchlist": str,                    # Optional: name of watchlist to screen
    "save_result": str                   # Optional: name to save results
}
```

### Category Values
Ava1ilable market cap categories for filtering:
- "mega_cap": >$200B
- "large_cap": $10B-$200B
- "mid_cap": $2B-$10B
- "small_cap": $300M-$2B
- "micro_cap": <$300M
- "etf": ETF instruments

2. `manage_watchlist`
```python
{
    "action": str,                       # Required: "create", "update", "delete", "get"
    "name": str,                         # Required: watchlist name (1-50 chars, alphanumeric with _ -)
    "symbols": List[str]                 # Required for create/update: list of stock symbols
}
```

3. `get_screening_result`
```python
{
    "name": str                          # Required: name of saved screening result
}
```

## Response Formats

### Technical Screen Response
```python
{
    "screen_type": "technical",
    "criteria": dict,                    # Original criteria used
    "matches": int,                      # Number of matching stocks
    "results": [                         # List of matching stocks
        {
            "symbol": str,
            "price": float,
            "volume": float,
            "rsi": float,
            "sma_20": float,
            "sma_50": float,
            "sma_200": float,
            "atr": float,
            "atr_pct": float,
            "price_changes": {
                "1d": float,             # 1-day price change %
                "5d": float,             # 5-day price change %
                "20d": float             # 20-day price change %
            },
            "ma_distances": {
                "pct_from_20sma": float,
                "pct_from_50sma": float,
                "pct_from_200sma": float
            }
        }
    ],
    "rejected": [                        # List of stocks that didn't match
        {
            "symbol": str,
            "rejection_reasons": List[str]
        }
    ],
    "timestamp": str
}
```
## Usage Prompt for Claude

"I've enabled the stockscreen tools which provide stock screening capabilities. You can use three main functions:

1. Screen stocks with various criteria types:
   - Technical: Price, volume, RSI, moving averages, ATR
   - Fundamental: Market cap, P/E, dividends, growth
   - Options: IV, volume, earnings dates
   - Custom: Combine multiple criteria types

2. Manage watchlists:
   - Create and update symbol lists
   - Delete existing watchlists
   - Retrieve watchlist contents

3. Access saved screening results:
   - Load previous screen results
   - Review matched symbols and criteria

All functions include error handling, detailed market data, and comprehensive responses."

## Requirements

- Python 3.8+
- MCP Server
- yfinance
- pandas
- numpy
- asyncio

## Limitations

- Data sourced from Yahoo Finance with potential delays
- Rate limits based on Yahoo Finance API restrictions
- Options data availability depends on market hours
- Some financial metrics may be delayed or unavailable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Todd Wolven - (https://github.com/twolven)

## Acknowledgments

- Built with the Model Context Protocol (MCP) by Anthropic
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Developed for use with Anthropic's Claude
