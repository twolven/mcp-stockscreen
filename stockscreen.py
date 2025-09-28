#!/usr/bin/env python3

import logging
import asyncio
import yfinance as yf
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
import json
import traceback
import pandas as pd
import datetime
from functools import wraps
import time
from typing import Optional, Dict, Any, List
import numpy as np
import os
from pathlib import Path

# Default to a data directory in current working directory if not specified
DEFAULT_DATA_PATH = os.environ.get('STOCKSCREEN_DATA_PATH', 
    os.path.join(os.path.dirname(__file__), "data"))
DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "stockscreen_v1.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DEFAULT_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stockscreen-server-v1")

class StockscreenError(Exception):
    pass

class ValidationError(StockscreenError):
    pass

class APIError(StockscreenError):
    pass

class DefaultSymbols:
    """Manages default symbol lists and market categories"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.cache_file = os.path.join(base_path, 'market_data', 'default_symbols.json')
        self.cache_expiry = 24 * 60 * 60  # 24 hours in seconds
        
    async def get_symbols(self, category: Optional[str] = None) -> List[str]:
        """
        Get default symbols, optionally filtered by category.
        Categories: mega_cap, large_cap, mid_cap, small_cap, micro_cap, etf
        """
        symbols_data = await self._load_or_fetch_symbols()
        
        if category:
            category = category.lower()
            if category not in self._get_category_filters():
                raise ValidationError(f"Invalid category: {category}")
            return self._filter_by_category(symbols_data, category)
        
        return [s['symbol'] for s in symbols_data]
    
    def _get_category_filters(self) -> Dict[str, Dict]:
        """Define market cap and other category filters"""
        return {
            'mega_cap': {'min_cap': 200e9},  # $200B+
            'large_cap': {'min_cap': 10e9, 'max_cap': 200e9},  # $10B-$200B
            'mid_cap': {'min_cap': 2e9, 'max_cap': 10e9},  # $2B-$10B
            'small_cap': {'min_cap': 300e6, 'max_cap': 2e9},  # $300M-$2B
            'micro_cap': {'max_cap': 300e6},  # Under $300M
            'etf': {'type': 'etf'}
        }
    
    def _filter_by_category(self, symbols_data: List[Dict], category: str) -> List[str]:
        """Filter symbols by category criteria"""
        filters = self._get_category_filters()[category]
        filtered = []
        
        for data in symbols_data:
            matches = True
            if 'type' in filters:
                if data.get('type') != filters['type']:
                    matches = False
            if 'min_cap' in filters and data.get('market_cap'):
                if data['market_cap'] < filters['min_cap']:
                    matches = False
            if 'max_cap' in filters and data.get('market_cap'):
                if data['market_cap'] > filters['max_cap']:
                    matches = False
            if matches:
                filtered.append(data['symbol'])
                
        return filtered
    
    async def _load_or_fetch_symbols(self) -> List[Dict]:
        """Load symbols from cache or fetch if expired"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cached = json.load(f)
                if time.time() - cached['timestamp'] < self.cache_expiry:
                    return cached['data']
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
        
        # Fetch fresh data
        data = await self._fetch_symbols()
        
        # Save to cache
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
        
        return data
    
    async def _fetch_symbols(self) -> List[Dict]:
        """Fetch symbols from major exchanges"""
        symbols_data = []
        
        # Helper function to fetch data for multiple symbols
        async def fetch_batch(symbols: List[str]) -> List[Dict]:
            batch_data = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info:
                        batch_data.append({
                            'symbol': symbol,
                            'type': info.get('quoteType', 'EQUITY').lower(),
                            'market_cap': info.get('marketCap'),
                            'exchange': info.get('exchange'),
                            'industry': info.get('industry'),
                            'sector': info.get('sector')
                        })
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {str(e)}")
            return batch_data
        
        try:
            # Get initial symbol list from yfinance
            # This uses top symbols from major exchanges
            major_symbols = []
            
            # Add major indices
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, Nasdaq
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    if hasattr(ticker, 'components'):
                        major_symbols.extend(ticker.components)
                except Exception as e:
                    logger.warning(f"Error getting components for {index}: {str(e)}")
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(major_symbols), batch_size):
                batch = major_symbols[i:i + batch_size]
                batch_data = await fetch_batch(batch)
                symbols_data.extend(batch_data)
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {str(e)}")
            raise APIError("Failed to fetch default symbols")
        
        return symbols_data

# Update the ScreenerDataStore class
class ScreenerDataStore:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self._ensure_directories()
        self.default_symbols = DefaultSymbols(base_path)

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}\n{traceback.format_exc()}")
            raise last_error
        return wrapper
    return decorator

class StockscreenJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

async def get_news_data(symbol: str, days_back: int = 30) -> dict:
    """
    Get recent news data for a symbol
    Returns dict with news items and key company events
    """
    try:
        ticker = yf.Ticker(symbol)
        news_data = {
            "recent_news": [],
            "key_events": [],
            "management_changes": [],
            "last_updated": datetime.datetime.now().isoformat()
        }

        # Get news from yfinance
        try:
            news = ticker.news
            if news:
                for item in news:
                    if (datetime.datetime.fromtimestamp(item['providerPublishTime']) > 
                        datetime.datetime.now() - datetime.timedelta(days=days_back)):
                        news_item = {
                            "title": item.get('title'),
                            "publisher": item.get('publisher'),
                            "published_at": datetime.datetime.fromtimestamp(item['providerPublishTime']).isoformat(),
                            "type": item.get('type'),
                            "summary": item.get('summary')
                        }
                        
                        # Categorize news
                        title_lower = news_item['title'].lower()
                        if any(term in title_lower for term in ['ceo', 'chief', 'executive', 'president', 'chairman']):
                            news_data['management_changes'].append(news_item)
                        elif any(term in title_lower for term in ['lawsuit', 'investigation', 'sec', 'probe']):
                            news_data['key_events'].append(news_item)
                        else:
                            news_data['recent_news'].append(news_item)

        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {str(e)}")
            
        # Additional data enrichment
        try:
            info = ticker.info
            if info:
                # Get executive team info
                if 'companyOfficers' in info:
                    news_data['current_management'] = [{
                        'name': officer.get('name'),
                        'title': officer.get('title'),
                        'since': officer.get('yearStarted')
                    } for officer in info['companyOfficers']]
                    
                # Get company description and updates
                news_data['company_info'] = {
                    'description': info.get('longBusinessSummary'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'website': info.get('website'),
                    'last_updated': datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.warning(f"Error fetching additional info for {symbol}: {str(e)}")

        return news_data
        
    except Exception as e:
        logger.error(f"Error in get_news_data for {symbol}: {str(e)}")
        return {
            "error": str(e),
            "last_updated": datetime.datetime.now().isoformat()
        }

def format_response(data: Any, error: Optional[str] = None) -> List[TextContent]:
    response = {
        "success": error is None,
        "timestamp": time.time(),
        "data": data if error is None else None,
        "error": error
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2, cls=StockscreenJSONEncoder)
    )]

# Add after the error classes
def validate_watchlist_name(name: str) -> bool:
    """
    Validate watchlist name:
    - Must be 1-50 characters
    - Can only contain letters, numbers, underscore, hyphen
    - Cannot start with hyphen
    """
    import re
    if not isinstance(name, str):
        raise ValidationError("Watchlist name must be a string")
    if not 1 <= len(name) <= 50:
        raise ValidationError("Watchlist name must be between 1 and 50 characters")
    if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_-]*$', name):
        raise ValidationError("Watchlist name can only contain letters, numbers, underscore, and hyphen")
    return True

def validate_stock_symbols(symbols: List[str], max_symbols: int = 1000) -> bool:
    """
    Validate stock symbols:
    - Must be list of strings
    - Each symbol must be valid format
    - Cannot exceed max_symbols
    """
    if not isinstance(symbols, list):
        raise ValidationError("Symbols must be a list")
    if len(symbols) > max_symbols:
        raise ValidationError(f"Cannot exceed {max_symbols} symbols")
    for symbol in symbols:
        if not isinstance(symbol, str):
            raise ValidationError("All symbols must be strings")
        if not 1 <= len(symbol) <= 10:
            raise ValidationError(f"Invalid symbol length: {symbol}")
        if not symbol.replace('-', '').replace('.', '').isalnum():
            raise ValidationError(f"Invalid symbol format: {symbol}")
    return True

# Data Storage Class
class ScreenerDataStore:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self._ensure_directories()
        self.default_symbols = DefaultSymbols(base_path)

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['screening_results', 'watchlists', 'market_data']
        for dir_name in directories:
            dir_path = os.path.join(self.base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

    def save_screening_result(self, name: str, data: dict):
        """Save screening result to JSON file"""
        file_path = os.path.join(self.base_path, 'screening_results', f'{name}.json')
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=StockscreenJSONEncoder)

    def load_screening_result(self, name: str) -> Optional[dict]:
        """Load screening result from JSON file"""
        file_path = os.path.join(self.base_path, 'screening_results', f'{name}.json')
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def save_watchlist(self, name: str, symbols: List[str]):
        """Save watchlist to JSON file"""
        file_path = os.path.join(self.base_path, 'watchlists', f'{name}.json')
        with open(file_path, 'w') as f:
            json.dump(symbols, f)

    def load_watchlist(self, name: str) -> Optional[List[str]]:
        """Load watchlist from JSON file"""
        file_path = os.path.join(self.base_path, 'watchlists', f'{name}.json')
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        
    # Add to ScreenerDataStore class
    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist file if it exists"""
        file_path = os.path.join(self.base_path, 'watchlists', f'{name}.json')
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False
        
# Initialize server and data store
app = Server("stockscreen-server-v1")
data_store = ScreenerDataStore(base_path=DEFAULT_DATA_PATH)

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="run_stock_screen",
            description="Screen stocks based on technical, fundamental, options, and news criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "screen_type": {
                        "type": "string",
                        "description": "Type of screen to run",
                        "enum": ["technical", "fundamental", "options", "news", "custom"]
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Screening criteria"
                    },
                    "watchlist": {
                        "type": "string",
                        "description": "Name of watchlist to screen (optional)"
                    },
                    "save_result": {
                        "type": "string",
                        "description": "Name to save screening result (optional)"
                    }
                },
                "required": ["screen_type", "criteria"]
            }
        ),
        Tool(
            name="get_stock_news",
            description="Get recent news and updates for a stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days of news to retrieve",
                        "default": 30
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="manage_watchlist",
            description="Create, update, or delete watchlists",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "delete", "get"],
                        "description": "Action to perform on watchlist"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of watchlist"
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols (for create/update)"
                    }
                },
                "required": ["action", "name"]
            }
        ),
        Tool(
            name="get_screening_result",
            description="Retrieve saved screening results",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of saved screening result"
                    }
                },
                "required": ["name"]
            }
        )
    ]

@app.call_tool()
@retry_on_error(max_retries=3, delay=1.0)
async def call_tool(name: str, arguments: dict):
    try:
        if name == "run_stock_screen":
            screen_type = arguments['screen_type']
            criteria = arguments['criteria']
            watchlist_name = arguments.get('watchlist')
            save_result = arguments.get('save_result')
            
            # Get symbols to screen
            symbols = None
            if watchlist_name:
                symbols = data_store.load_watchlist(watchlist_name)
                if not symbols:
                    raise ValidationError(f"Watchlist '{watchlist_name}' not found")
                
            # Run appropriate screen based on type
            if screen_type == "technical":
                result = await run_technical_screen(symbols, criteria)
            elif screen_type == "fundamental":
                result = await run_fundamental_screen(symbols, criteria)
            elif screen_type == "options":
                result = await run_options_screen(symbols, criteria)
            elif screen_type == "news":
                result = await run_news_screen(symbols, criteria)
            elif screen_type == "custom":
                result = await run_custom_screen(symbols, criteria)
            else:
                raise ValidationError(f"Invalid screen type: {screen_type}")
                
            # Save result if requested
            if save_result:
                data_store.save_screening_result(save_result, result)
                
            return format_response(result)
            
        elif name == "get_stock_news":
            symbol = arguments['symbol']
            days_back = arguments.get('days_back', 30)
            news_data = await get_news_data(symbol, days_back)
            return format_response(news_data)
            
        elif name == "manage_watchlist":
            action = arguments['action']
            watchlist_name = arguments['name']
            
            # Validate watchlist name
            validate_watchlist_name(watchlist_name)
            
            if action == "create" or action == "update":
                if 'symbols' not in arguments:
                    raise ValidationError("symbols required for create/update")
                symbols = arguments['symbols']
                # Validate symbols
                validate_stock_symbols(symbols)
                data_store.save_watchlist(watchlist_name, symbols)
                return format_response({"message": f"Watchlist '{watchlist_name}' saved with {len(symbols)} symbols"})
                
            elif action == "delete":
                if data_store.delete_watchlist(watchlist_name):
                    return format_response({"message": f"Watchlist '{watchlist_name}' deleted"})
                else:
                    raise ValidationError(f"Watchlist '{watchlist_name}' not found")
                
            elif action == "get":
                symbols = data_store.load_watchlist(watchlist_name)
                if symbols is None:
                    raise ValidationError(f"Watchlist '{watchlist_name}' not found")
                return format_response({"name": watchlist_name, "symbols": symbols})
                
            else:
                raise ValidationError(f"Invalid action: {action}")
                
        elif name == "get_screening_result":
            result_name = arguments['name']
            result = data_store.load_screening_result(result_name)
            if result is None:
                raise ValidationError(f"Screening result '{result_name}' not found")
            return format_response(result)
            
    except ValidationError as e:
        logger.error(f"Validation error in {name}: {str(e)}")
        return format_response(None, f"Validation error: {str(e)}")
        
    except APIError as e:
        logger.error(f"API error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"API error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"Internal error: {str(e)}")

# Screen implementation functions
async def run_technical_screen(symbols: Optional[List[str]], criteria: dict) -> dict:
    """Run enhanced technical analysis based screen"""
    if not symbols:
        if 'symbols' in criteria:
            symbols = [criteria['symbols']] if isinstance(criteria['symbols'], str) else criteria['symbols']
        else:
            # Get default symbols, optionally filtered by category
            category = criteria.get('category')
            symbols = await data_store.default_symbols.get_symbols(category)
            
    results = []
    rejected_results = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data with better error handling
            try:
                hist = ticker.history(period="1y")  # Get 1 year for better SMA200 calculation
                if hist.empty:
                    raise ValueError("No historical data available")
            except Exception as e:
                logger.error(f"Error getting history for {symbol}: {str(e)}")
                rejected_results.append({
                    "symbol": symbol,
                    "error": f"Historical data error: {str(e)}"
                })
                continue
                
            # Calculate technical indicators
            data = hist.copy()
            rejection_reasons = []
            
            # Get current price with error handling
            try:
                current_price = (
                    ticker.info.get('regularMarketPrice') or 
                    ticker.info.get('currentPrice') or 
                    data['Close'].iloc[-1]
                )
                if pd.isna(current_price):
                    raise ValueError("No valid price data")
            except Exception as e:
                rejection_reasons.append(f"Price data error: {str(e)}")
                continue
                
            # Price criteria
            if 'min_price' in criteria and current_price < criteria['min_price']:
                rejection_reasons.append(f"Price ({current_price:.2f}) < minimum ({criteria['min_price']})")
            if 'max_price' in criteria and current_price > criteria['max_price']:
                rejection_reasons.append(f"Price ({current_price:.2f}) > maximum ({criteria['max_price']})")
                
            # Volume with error handling
            try:
                avg_volume = data['Volume'].mean()
                if pd.isna(avg_volume):
                    raise ValueError("No valid volume data")
                if 'min_volume' in criteria and avg_volume < criteria['min_volume']:
                    rejection_reasons.append(f"Volume ({avg_volume:.0f}) < minimum ({criteria['min_volume']})")
            except Exception as e:
                rejection_reasons.append(f"Volume calculation error: {str(e)}")
                
            # Moving averages with error handling
            try:
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()
                
                # Get latest values with NaN handling
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                sma_200 = data['SMA_200'].iloc[-1]
                
                # Check MA criteria
                if criteria.get('above_sma_200', False):
                    if pd.isna(sma_200):
                        rejection_reasons.append("Insufficient data for SMA200 calculation")
                    elif current_price <= sma_200:
                        rejection_reasons.append(f"Price ({current_price:.2f}) below SMA200 ({sma_200:.2f})")
                        
                if criteria.get('above_sma_50', False):
                    if pd.isna(sma_50):
                        rejection_reasons.append("Insufficient data for SMA50 calculation")
                    elif current_price <= sma_50:
                        rejection_reasons.append(f"Price ({current_price:.2f}) below SMA50 ({sma_50:.2f})")
                        
            except Exception as e:
                rejection_reasons.append(f"Moving average calculation error: {str(e)}")
                
            # RSI calculation with error handling
            try:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                current_rsi = data['RSI'].iloc[-1]
                if pd.isna(current_rsi):
                    raise ValueError("RSI calculation resulted in NaN")
                    
                if 'min_rsi' in criteria and current_rsi < criteria['min_rsi']:
                    rejection_reasons.append(f"RSI ({current_rsi:.1f}) < minimum ({criteria['min_rsi']})")
                if 'max_rsi' in criteria and current_rsi > criteria['max_rsi']:
                    rejection_reasons.append(f"RSI ({current_rsi:.1f}) > maximum ({criteria['max_rsi']})")
                    
            except Exception as e:
                rejection_reasons.append(f"RSI calculation error: {str(e)}")
                current_rsi = None
                
            # Volatility (ATR) calculation with error handling
            try:
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                data['ATR'] = true_range.rolling(window=14).mean()
                current_atr = data['ATR'].iloc[-1]
                
                if pd.isna(current_atr):
                    raise ValueError("ATR calculation resulted in NaN")
                    
                # Optional ATR criteria
                if 'max_atr_pct' in criteria:
                    atr_pct = (current_atr / current_price) * 100
                    if atr_pct > criteria['max_atr_pct']:
                        rejection_reasons.append(f"ATR% ({atr_pct:.1f}%) > maximum ({criteria['max_atr_pct']}%)")
                        
            except Exception as e:
                rejection_reasons.append(f"ATR calculation error: {str(e)}")
                current_atr = None
                
            # Add trending indicators
            try:
                # Calculate price changes for different timeframes
                data['1d_change'] = data['Close'].pct_change()
                data['5d_change'] = data['Close'].pct_change(periods=5)
                data['20d_change'] = data['Close'].pct_change(periods=20)
                
                changes = {
                    '1d': data['1d_change'].iloc[-1] * 100,
                    '5d': data['5d_change'].iloc[-1] * 100,
                    '20d': data['20d_change'].iloc[-1] * 100
                }
                
                # Check if any values are NaN
                for period, change in changes.items():
                    if pd.isna(change):
                        changes[period] = None
                        
            except Exception as e:
                logger.error(f"Error calculating price changes for {symbol}: {str(e)}")
                changes = {'1d': None, '5d': None, '20d': None}
                
            # If we have rejection reasons, add to rejected results
            if rejection_reasons:
                rejected_results.append({
                    "symbol": symbol,
                    "price": current_price,
                    "volume": avg_volume,
                    "rsi": current_rsi,
                    "sma_20": None if pd.isna(sma_20) else sma_20,
                    "sma_50": None if pd.isna(sma_50) else sma_50,
                    "sma_200": None if pd.isna(sma_200) else sma_200,
                    "atr": current_atr,
                    "price_changes": changes,
                    "rejection_reasons": rejection_reasons
                })
                continue
                
            # If we made it here, add to successful results
            results.append({
                "symbol": symbol,
                "price": current_price,
                "volume": avg_volume,
                "rsi": current_rsi,
                "sma_20": None if pd.isna(sma_20) else sma_20,
                "sma_50": None if pd.isna(sma_50) else sma_50,
                "sma_200": None if pd.isna(sma_200) else sma_200,
                "atr": current_atr,
                "atr_pct": (current_atr / current_price) * 100 if current_atr else None,
                "price_changes": changes,
                "ma_distances": {
                    "pct_from_20sma": ((current_price / sma_20) - 1) * 100 if not pd.isna(sma_20) else None,
                    "pct_from_50sma": ((current_price / sma_50) - 1) * 100 if not pd.isna(sma_50) else None,
                    "pct_from_200sma": ((current_price / sma_200) - 1) * 100 if not pd.isna(sma_200) else None
                }
            })
            
        except Exception as e:
            logger.error(f"Error screening {symbol}: {str(e)}")
            rejected_results.append({
                "symbol": symbol,
                "error": str(e)
            })
            continue
            
    return {
        "screen_type": "technical",
        "criteria": criteria,
        "matches": len(results),
        "results": results,
        "rejected": rejected_results,
        "timestamp": datetime.datetime.now().isoformat()
    }

async def run_fundamental_screen(symbols: Optional[List[str]], criteria: dict) -> dict:
    """Run fundamental analysis based screen"""
    if not symbols:
        if 'symbols' in criteria:
            symbols = [criteria['symbols']] if isinstance(criteria['symbols'], str) else criteria['symbols']
        else:
            # Get default symbols, optionally filtered by category
            category = criteria.get('category')
            symbols = await data_store.default_symbols.get_symbols(category)
    results = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Near top of run_fundamental_screen()
            info = ticker.info
            if not info:
                logger.warning(f"No fundamental data available for {symbol}")
                continue

            # For ETFs, only check some metrics
            is_etf = info.get('quoteType') == 'ETF'
            if is_etf:
                # Get ETF specific metrics
                current_price = (
                    info.get('regularMarketPrice') or 
                    info.get('previousClose') or
                    ticker.history(period="1d")['Close'].iloc[-1]
                )
                
                volume = (
                    info.get('regularMarketVolume') or 
                    info.get('averageVolume') or 
                    ticker.history(period="1mo")['Volume'].mean()
                )
                
                aum = info.get('totalAssets', 0)
                expense_ratio = info.get('expenseRatio', 0)
                
                # Apply ETF specific criteria
                if 'min_aum' in criteria and aum < criteria['min_aum']:
                    continue
                if 'max_expense_ratio' in criteria and expense_ratio > criteria['max_expense_ratio']:
                    continue
                if 'min_volume' in criteria and volume < criteria['min_volume']:
                    continue
                    
                # Store ETF results
                results.append({
                    "symbol": symbol,
                    "price": current_price,
                    "aum": aum,
                    "expense_ratio": expense_ratio,
                    "average_volume": volume,
                    "category": info.get('category', 'Unknown'),
                    "asset_class": info.get('assetClass', 'Unknown')
                })
                continue  # Skip regular stock metrics
                
            # Market cap criteria
            market_cap = info.get('marketCap', 0)
            if 'min_market_cap' in criteria and market_cap < criteria['min_market_cap']:
                continue
                
            # P/E criteria
            pe_ratio = info.get('forwardPE', 0)
            if 'min_pe' in criteria and pe_ratio < criteria['min_pe']:
                continue
            if 'max_pe' in criteria and pe_ratio > criteria['max_pe']:
                continue
                
            # Dividend criteria
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield *= 100  # Convert to percentage
            if 'min_dividend' in criteria and dividend_yield < criteria['min_dividend']:
                continue
                
            # Growth criteria
            revenue_growth = info.get('revenueGrowth', 0)
            if 'min_revenue_growth' in criteria and revenue_growth < criteria['min_revenue_growth']:
                continue
                
            # Store results
            results.append({
                "symbol": symbol,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield,
                "revenue_growth": revenue_growth,
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown')
            })
            
        except Exception as e:
            logger.error(f"Error screening {symbol}: {str(e)}")
            continue
            
    return {
        "screen_type": "fundamental",
        "criteria": criteria,
        "matches": len(results),
        "results": results,
        "timestamp": datetime.datetime.now().isoformat()
    }

async def get_earnings_dates(ticker: yf.Ticker) -> dict:
    """Get earnings dates handling yfinance quirks"""
    try:
        calendar = ticker.calendar
        next_dates = []
        
        if calendar is not None and isinstance(calendar, dict):
            earnings_date = calendar.get('Earnings Date')
            if isinstance(earnings_date, list):
                next_dates = earnings_date  # This will be a list of datetime.date objects
        
        days_to_earnings = None
        if next_dates:
            earliest_date = next_dates[0]  # Will be a datetime.date object
            days_to_earnings = (earliest_date - datetime.now().date()).days
            
        return {
            "next_earnings": next_dates[0] if next_dates else None,
            "earnings_range_end": next_dates[1] if len(next_dates) > 1 else next_dates[0] if next_dates else None,
            "days_to_earnings": days_to_earnings,
            "is_estimate": len(next_dates) > 1 if next_dates else None
        }
    except Exception as e:
        logger.error(f"Error getting earnings dates: {str(e)}")
        return {
            "next_earnings": None,
            "earnings_range_end": None,
            "days_to_earnings": None,
            "is_estimate": None
        }

async def run_options_screen(symbols: Optional[List[str]], criteria: dict) -> dict:
    """Run options based screen with enhanced data"""
    if not symbols:
        if 'symbols' in criteria:
            symbols = [criteria['symbols']] if isinstance(criteria['symbols'], str) else criteria['symbols']
        else:
            # Get default symbols, optionally filtered by category
            category = criteria.get('category')
            symbols = await data_store.default_symbols.get_symbols(category)  
    
    results = []
    rejected_results = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings data
            earnings_info = await get_earnings_dates(ticker)
            
            # Track rejection reasons
            rejection_reasons = []
            
            # Check earnings criteria if specified
            earnings_valid = True
            if earnings_info and 'min_days_to_earnings' in criteria:
                if (earnings_info['days_to_earnings'] is None or 
                    earnings_info['days_to_earnings'] < criteria['min_days_to_earnings']):
                    earnings_valid = False
                    rejection_reasons.append(f"Days to earnings ({earnings_info['days_to_earnings']}) < minimum ({criteria['min_days_to_earnings']})")
                    
            if earnings_info and 'max_days_to_earnings' in criteria:
                if (earnings_info['days_to_earnings'] is None or 
                    earnings_info['days_to_earnings'] > criteria['max_days_to_earnings']):
                    earnings_valid = False
                    rejection_reasons.append(f"Days to earnings ({earnings_info['days_to_earnings']}) > maximum ({criteria['max_days_to_earnings']})")
            
            # Get current price with better error handling
            try:
                current_price = (
                    ticker.info.get('regularMarketPrice') or 
                    ticker.info.get('currentPrice') or 
                    ticker.history(period="1d")['Close'].iloc[-1]
                )
                if not current_price or pd.isna(current_price):
                    raise ValueError("No valid price data available")
            except Exception as e:
                rejection_reasons.append(f"Price data error: {str(e)}")
                current_price = None
                
            # Get options expiration dates
            try:
                exp_dates = ticker.options
                if not exp_dates:
                    raise ValueError("No options expiration dates available")
            except Exception as e:
                rejection_reasons.append(f"Options data error: {str(e)}")
                continue
                
            # Enhanced options data collection
            options_data = {}
            try:
                # Get chains for next few expirations
                for exp_date in exp_dates[:3]:  # Look at first 3 expiration dates
                    chain = ticker.option_chain(exp_date)
                    if chain and hasattr(chain, 'calls') and hasattr(chain, 'puts'):
                        calls = chain.calls
                        puts = chain.puts
                        
                        # Calculate spreads and liquidity metrics
                        calls['bid_ask_spread'] = (calls['ask'] - calls['bid']) / calls['ask'] * 100
                        puts['bid_ask_spread'] = (puts['ask'] - puts['bid']) / puts['ask'] * 100
                        
                        options_data[exp_date] = {
                            'calls_count': len(calls),
                            'puts_count': len(puts),
                            'call_strikes': sorted(calls['strike'].unique().tolist()),
                            'put_strikes': sorted(puts['strike'].unique().tolist()),
                            'avg_call_spread': calls['bid_ask_spread'].mean(),
                            'avg_put_spread': puts['bid_ask_spread'].mean(),
                            'total_call_volume': int(calls['volume'].sum()),
                            'total_put_volume': int(puts['volume'].sum()),
                            'total_call_oi': int(calls['openInterest'].sum()),
                            'total_put_oi': int(puts['openInterest'].sum())
                        }
            except Exception as e:
                logger.error(f"Error processing options chains for {symbol}: {str(e)}")
                options_data['error'] = str(e)

            # If we have rejection reasons, add to rejected results with enhanced data
            if rejection_reasons:
                rejected_results.append({
                    "symbol": symbol,
                    "price": current_price,
                    "next_earnings": earnings_info['next_earnings'] if earnings_info else None,
                    "days_to_earnings": earnings_info['days_to_earnings'] if earnings_info else None,
                    "options_data": options_data,
                    "rejection_reasons": rejection_reasons
                })
                continue

            # Get first expiration chain for standard metrics
            try:
                options = ticker.option_chain(exp_dates[0])
                if not options or not hasattr(options, 'calls') or not hasattr(options, 'puts'):
                    raise ValueError("Invalid options chain data")
                    
                # Find ATM strike
                atm_strike = min(options.calls['strike'], key=lambda x: abs(x - current_price))
                
                # Get ATM options with error checking
                atm_calls = options.calls[options.calls['strike'] == atm_strike]
                atm_puts = options.puts[options.puts['strike'] == atm_strike]
                
                if atm_calls.empty or atm_puts.empty:
                    raise ValueError("No ATM options data available")
                    
                # Calculate metrics
                atm_call_iv = float(atm_calls['impliedVolatility'].iloc[0]) * 100
                atm_put_iv = float(atm_puts['impliedVolatility'].iloc[0]) * 100
                avg_iv = (atm_call_iv + atm_put_iv) / 2
                
                # Calculate bid-ask spreads
                atm_call_spread = (float(atm_calls['ask'].iloc[0]) - float(atm_calls['bid'].iloc[0])) / float(atm_calls['ask'].iloc[0]) * 100
                atm_put_spread = (float(atm_puts['ask'].iloc[0]) - float(atm_puts['bid'].iloc[0])) / float(atm_puts['ask'].iloc[0]) * 100
                
                # IV criteria check
                if 'min_iv' in criteria and avg_iv < criteria['min_iv']:
                    rejection_reasons.append(f"IV ({avg_iv:.1f}%) < minimum ({criteria['min_iv']}%)")
                    
                if 'max_iv' in criteria and avg_iv > criteria['max_iv']:
                    rejection_reasons.append(f"IV ({avg_iv:.1f}%) > maximum ({criteria['max_iv']}%)")
                
                # Volume calculations
                total_volume = int(options.calls['volume'].sum() + options.puts['volume'].sum())
                total_oi = int(options.calls['openInterest'].sum() + options.puts['openInterest'].sum())
                
                if 'min_option_volume' in criteria and total_volume < criteria['min_option_volume']:
                    rejection_reasons.append(f"Option volume ({total_volume}) < minimum ({criteria['min_option_volume']})")
                
                # Put/Call ratio check
                put_volume = options.puts['volume'].sum()
                call_volume = options.calls['volume'].sum()
                put_call_ratio = put_volume / max(1, call_volume)
                
                if 'min_put_call_ratio' in criteria and put_call_ratio < criteria['min_put_call_ratio']:
                    rejection_reasons.append(f"Put/Call ratio ({put_call_ratio:.2f}) < minimum ({criteria['min_put_call_ratio']})")
                
                # Spread checks
                avg_spread = (atm_call_spread + atm_put_spread) / 2
                if 'max_spread' in criteria and avg_spread > criteria['max_spread']:
                    rejection_reasons.append(f"Avg spread ({avg_spread:.1f}%) > maximum ({criteria['max_spread']}%)")
                
                # If we accumulated any rejection reasons after full analysis
                if rejection_reasons:
                    rejected_results.append({
                        "symbol": symbol,
                        "price": current_price,
                        "implied_volatility": avg_iv / 100,  # Store as decimal
                        "implied_volatility_pct": avg_iv,    # Store as percentage
                        "option_volume": total_volume,
                        "put_call_ratio": put_call_ratio,
                        "next_earnings": earnings_info['next_earnings'] if earnings_info else None,
                        "days_to_earnings": earnings_info['days_to_earnings'] if earnings_info else None,
                        "options_data": options_data,
                        "atm_metrics": {
                            "strike": atm_strike,
                            "call_spread": atm_call_spread,
                            "put_spread": atm_put_spread,
                            "avg_spread": avg_spread
                        },
                        "rejection_reasons": rejection_reasons
                    })
                    continue
                    
                # If we made it here, add to successful results
                results.append({
                    "symbol": symbol,
                    "price": current_price,
                    "atm_strike": atm_strike,
                    "implied_volatility": avg_iv / 100,
                    "implied_volatility_pct": avg_iv,
                    "call_iv": atm_call_iv,
                    "put_iv": atm_put_iv,
                    "option_volume": total_volume,
                    "open_interest": total_oi,
                    "put_call_ratio": put_call_ratio,
                    "nearest_expiry": exp_dates[0],
                    "next_earnings": earnings_info['next_earnings'] if earnings_info else None,
                    "days_to_earnings": earnings_info['days_to_earnings'] if earnings_info else None,
                    "options_data": options_data,
                    "atm_metrics": {
                        "strike": atm_strike,
                        "call_spread": atm_call_spread,
                        "put_spread": atm_put_spread,
                        "avg_spread": avg_spread,
                        "call_bid": float(atm_calls['bid'].iloc[0]),
                        "call_ask": float(atm_calls['ask'].iloc[0]),
                        "put_bid": float(atm_puts['bid'].iloc[0]),
                        "put_ask": float(atm_puts['ask'].iloc[0])
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing ATM options for {symbol}: {str(e)}")
                rejection_reasons.append(f"ATM options error: {str(e)}")
                continue
                
        except Exception as e:
            logger.error(f"Error screening {symbol}: {str(e)}")
            rejected_results.append({
                "symbol": symbol,
                "error": str(e)
            })
            continue
            
    return {
        "screen_type": "options",
        "criteria": criteria,
        "matches": len(results),
        "results": results,
        "rejected": rejected_results,
        "timestamp": datetime.datetime.now().isoformat()
    }

async def run_news_screen(symbols: Optional[List[str]], criteria: dict) -> dict:
    """
    Screen stocks based on news and events criteria
    
    Example criteria:
    {
        "keywords": ["acquisition", "merger"],
        "exclude_keywords": ["lawsuit", "investigation"],
        "min_days": 1,
        "max_days": 30,
        "management_changes": true,
        "require_all_keywords": false
    }
    """
    if not symbols:
        if 'symbols' in criteria:
            symbols = [criteria['symbols']] if isinstance(criteria['symbols'], str) else criteria['symbols']
        else:
            # Get default symbols, optionally filtered by category
            category = criteria.get('category')
            symbols = await data_store.default_symbols.get_symbols(category)
            
    results = []
    rejected = []
    
    for symbol in symbols:
        try:
            news_data = await get_news_data(symbol, days_back=criteria.get('max_days', 30))
            
            if 'error' in news_data:
                rejected.append({
                    "symbol": symbol,
                    "error": news_data['error']
                })
                continue
                
            # Combine all news for searching
            all_news = (
                news_data.get('recent_news', []) + 
                news_data.get('key_events', []) + 
                news_data.get('management_changes', [])
            )
            
            # Filter by date range
            min_date = datetime.datetime.now() - datetime.timedelta(days=criteria.get('max_days', 30))
            max_date = datetime.datetime.now() - datetime.timedelta(days=criteria.get('min_days', 0))
            
            filtered_news = [
                news for news in all_news
                if min_date <= datetime.datetime.fromisoformat(news['published_at']) <= max_date
            ]
            
            # Check keywords
            keywords = criteria.get('keywords', [])
            exclude_keywords = criteria.get('exclude_keywords', [])
            require_all = criteria.get('require_all_keywords', False)
            
            matching_news = []
            for news in filtered_news:
                text = f"{news['title']} {news['summary']}".lower()
                
                # Check excluded keywords first
                if any(kw.lower() in text for kw in exclude_keywords):
                    continue
                    
                # Check included keywords
                if keywords:
                    if require_all:
                        if all(kw.lower() in text for kw in keywords):
                            matching_news.append(news)
                    else:
                        if any(kw.lower() in text for kw in keywords):
                            matching_news.append(news)
                else:
                    matching_news.append(news)
                    
            # Check management changes if requested
            if criteria.get('management_changes'):
                if not news_data.get('management_changes'):
                    rejected.append({
                        "symbol": symbol,
                        "reason": "No recent management changes found"
                    })
                    continue
                    
            # Add to results if we found matching news
            if matching_news:
                results.append({
                    "symbol": symbol,
                    "matching_news": matching_news,
                    "management": news_data.get('current_management'),
                    "company_info": news_data.get('company_info')
                })
            else:
                rejected.append({
                    "symbol": symbol,
                    "reason": "No matching news found"
                })
                
        except Exception as e:
            logger.error(f"Error screening news for {symbol}: {str(e)}")
            rejected.append({
                "symbol": symbol,
                "error": str(e)
            })
            
    return {
        "screen_type": "news",
        "criteria": criteria,
        "matches": len(results),
        "results": results,
        "rejected": rejected,
        "timestamp": datetime.datetime.now().isoformat()
    }

async def run_custom_screen(symbols: Optional[List[str]], criteria: dict) -> dict:
    """
    Run custom combination screen that can mix technical, fundamental, options, and news criteria
    
    Example criteria:
    {
        "technical": {
            "min_price": 20,
            "above_sma_200": true
        },
        "fundamental": {
            "min_market_cap": 10000000000
        },
        "options": {
            "min_option_volume": 10000
        },
        "news": {
            "keywords": ["acquisition"],
            "max_days": 30,
            "management_changes": true
        }
    }
    """
    try:
        if not symbols:
            if 'symbols' in criteria:
                symbols = [criteria['symbols']] if isinstance(criteria['symbols'], str) else criteria['symbols']
            else:
                category = criteria.get('category')
                symbols = await data_store.default_symbols.get_symbols(category)

        results = []
        rejected_results = []

        # Extract criteria sections
        technical_criteria = criteria.get('technical', {})
        fundamental_criteria = criteria.get('fundamental', {})
        options_criteria = criteria.get('options', {})
        news_criteria = criteria.get('news', {})

        # Process each symbol
        for symbol in symbols:
            try:
                rejection_reasons = []
                symbol_data = {'symbol': symbol}

                # Run each screen type if criteria provided
                if technical_criteria:
                    tech_result = await run_single_technical_screen(symbol, technical_criteria)
                    if tech_result.get('rejection_reasons'):
                        rejection_reasons.extend(tech_result['rejection_reasons'])
                    symbol_data.update(tech_result.get('data', {}))

                if fundamental_criteria and not rejection_reasons:
                    fund_result = await run_single_fundamental_screen(symbol, fundamental_criteria)
                    if fund_result.get('rejection_reasons'):
                        rejection_reasons.extend(fund_result['rejection_reasons'])
                    symbol_data.update(fund_result.get('data', {}))

                if options_criteria and not rejection_reasons:
                    opt_result = await run_single_options_screen(symbol, options_criteria)
                    if opt_result.get('rejection_reasons'):
                        rejection_reasons.extend(opt_result['rejection_reasons'])
                    symbol_data.update(opt_result.get('data', {}))

                # Add news screening
                if news_criteria and not rejection_reasons:
                    news_result = await get_news_data(symbol, days_back=news_criteria.get('max_days', 30))
                    if 'error' in news_result:
                        rejection_reasons.append(f"News error: {news_result['error']}")
                    else:
                        # Apply news criteria
                        matching_news = []
                        all_news = (
                            news_result.get('recent_news', []) + 
                            news_result.get('key_events', []) + 
                            news_result.get('management_changes', [])
                        )
                        
                        for news in all_news:
                            text = f"{news['title']} {news['summary']}".lower()
                            keywords = news_criteria.get('keywords', [])
                            if keywords and not any(kw.lower() in text for kw in keywords):
                                continue
                            matching_news.append(news)
                            
                        if news_criteria.get('management_changes') and not news_result.get('management_changes'):
                            rejection_reasons.append("No recent management changes found")
                        elif matching_news:
                            symbol_data['news'] = matching_news
                            symbol_data['management'] = news_result.get('current_management')
                        else:
                            rejection_reasons.append("No matching news found")

                # Add to appropriate results list
                if rejection_reasons:
                    symbol_data['rejection_reasons'] = rejection_reasons
                    rejected_results.append(symbol_data)
                else:
                    results.append(symbol_data)

            except Exception as e:
                logger.error(f"Error screening {symbol}: {str(e)}")
                rejected_results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
                continue

        return {
            "screen_type": "custom",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected_results,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in custom screen: {str(e)}\n{traceback.format_exc()}")
        raise APIError(f"Custom screen failed: {str(e)}")

# Helper functions for single-symbol screening
async def run_single_technical_screen(symbol: str, criteria: dict) -> dict:
    """Run technical screen on a single symbol"""
    ticker = yf.Ticker(symbol)
    rejection_reasons = []
    data = {}
    
    try:
        # Get current price
        current_price = (
            ticker.info.get('regularMarketPrice') or 
            ticker.info.get('currentPrice') or 
            ticker.history(period="1d")['Close'].iloc[-1]
        )
        
        if pd.isna(current_price):
            raise ValueError("No valid price data")
        
        data['price'] = current_price
        
        # Get historical data for technicals
        hist = ticker.history(period="1y")
        if hist.empty:
            raise ValueError("No historical data available")
            
        # Calculate technical indicators
        hist_data = hist.copy()
        
        # Volume check
        avg_volume = hist_data['Volume'].mean()
        data['volume'] = avg_volume
        if 'min_volume' in criteria and avg_volume < criteria['min_volume']:
            rejection_reasons.append(f"Volume ({avg_volume:.0f}) < minimum ({criteria['min_volume']})")
            
        # Price criteria
        if 'min_price' in criteria and current_price < criteria['min_price']:
            rejection_reasons.append(f"Price ({current_price:.2f}) < minimum ({criteria['min_price']})")
        if 'max_price' in criteria and current_price > criteria['max_price']:
            rejection_reasons.append(f"Price ({current_price:.2f}) > maximum ({criteria['max_price']})")
            
        # Moving averages
        if 'above_sma_200' in criteria or 'above_sma_50' in criteria:
            hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            hist_data['SMA_200'] = hist_data['Close'].rolling(window=200).mean()
            
            sma_50 = hist_data['SMA_50'].iloc[-1]
            sma_200 = hist_data['SMA_200'].iloc[-1]
            
            data['sma_50'] = sma_50
            data['sma_200'] = sma_200
            
            if criteria.get('above_sma_200', False) and current_price <= sma_200:
                rejection_reasons.append(f"Price ({current_price:.2f}) below SMA200 ({sma_200:.2f})")
            if criteria.get('above_sma_50', False) and current_price <= sma_50:
                rejection_reasons.append(f"Price ({current_price:.2f}) below SMA50 ({sma_50:.2f})")
                
        return {
            "data": data,
            "rejection_reasons": rejection_reasons
        }
                
    except Exception as e:
        return {
            "data": {},
            "rejection_reasons": [f"Technical analysis error: {str(e)}"]
        }

async def run_single_fundamental_screen(symbol: str, criteria: dict) -> dict:
    """Run fundamental screen on a single symbol"""
    ticker = yf.Ticker(symbol)
    rejection_reasons = []
    data = {}
    
    try:
        info = ticker.info
        if not info:
            raise ValueError("No fundamental data available")
            
        # Market cap check
        market_cap = info.get('marketCap', 0)
        data['market_cap'] = market_cap
        if 'min_market_cap' in criteria and market_cap < criteria['min_market_cap']:
            rejection_reasons.append(f"Market cap ({market_cap}) < minimum ({criteria['min_market_cap']})")
            
        # P/E check
        pe_ratio = info.get('forwardPE', 0)
        data['pe_ratio'] = pe_ratio
        if 'min_pe' in criteria and pe_ratio < criteria['min_pe']:
            rejection_reasons.append(f"P/E ({pe_ratio:.1f}) < minimum ({criteria['min_pe']})")
        if 'max_pe' in criteria and pe_ratio > criteria['max_pe']:
            rejection_reasons.append(f"P/E ({pe_ratio:.1f}) > maximum ({criteria['max_pe']})")
            
        return {
            "data": data,
            "rejection_reasons": rejection_reasons
        }
        
    except Exception as e:
        return {
            "data": {},
            "rejection_reasons": [f"Fundamental analysis error: {str(e)}"]
        }

async def run_single_options_screen(symbol: str, criteria: dict) -> dict:
    """Run options screen on a single symbol"""
    ticker = yf.Ticker(symbol)
    rejection_reasons = []
    data = {}
    
    try:
        # Get options expiration dates
        exp_dates = ticker.options
        if not exp_dates:
            raise ValueError("No options data available")
            
        # Get first expiration chain for standard metrics
        options = ticker.option_chain(exp_dates[0])
        if not options or not hasattr(options, 'calls') or not hasattr(options, 'puts'):
            raise ValueError("Invalid options chain data")
            
        # Calculate options metrics
        total_volume = int(options.calls['volume'].sum() + options.puts['volume'].sum())
        data['option_volume'] = total_volume
        if 'min_option_volume' in criteria and total_volume < criteria['min_option_volume']:
            rejection_reasons.append(f"Option volume ({total_volume}) < minimum ({criteria['min_option_volume']})")
            
        # Get earnings data if needed
        if 'min_days_to_earnings' in criteria or 'max_days_to_earnings' in criteria:
            earnings_info = await get_earnings_dates(ticker)
            days_to_earnings = earnings_info['days_to_earnings']
            data['days_to_earnings'] = days_to_earnings
            
            if 'min_days_to_earnings' in criteria and days_to_earnings < criteria['min_days_to_earnings']:
                rejection_reasons.append(f"Days to earnings ({days_to_earnings}) < minimum ({criteria['min_days_to_earnings']})")
                
            if 'max_days_to_earnings' in criteria and days_to_earnings > criteria['max_days_to_earnings']:
                rejection_reasons.append(f"Days to earnings ({days_to_earnings}) > maximum ({criteria['max_days_to_earnings']})")
                
        return {
            "data": data,
            "rejection_reasons": rejection_reasons
        }
        
    except Exception as e:
        return {
            "data": {},
            "rejection_reasons": [f"Options analysis error: {str(e)}"]
        }

async def main():    
    logger.info("Starting Stockscreen server v1...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())