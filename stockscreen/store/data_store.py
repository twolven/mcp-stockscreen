"""Data persistence for watchlists, screening results, and default symbols."""

import json
import logging
import os
import time
from typing import Optional

from stockscreen.exceptions import APIError, ValidationError
from stockscreen.models.schemas import StockscreenJSONEncoder

logger = logging.getLogger("stockscreen-server-v1")


class ScreenerDataStore:
    """JSON file-based persistence for watchlists and screening results."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self._ensure_directories()
        self.default_symbols = DefaultSymbols(base_path)

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_name in ("screening_results", "watchlists", "market_data"):
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)

    def save_watchlist(self, name: str, symbols: list[str]):
        """Save watchlist to JSON file."""
        file_path = os.path.join(self.base_path, "watchlists", f"{name}.json")
        with open(file_path, "w") as f:
            json.dump(symbols, f)

    def load_watchlist(self, name: str) -> Optional[list[str]]:
        """Load watchlist from JSON file."""
        file_path = os.path.join(self.base_path, "watchlists", f"{name}.json")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist file if it exists."""
        file_path = os.path.join(self.base_path, "watchlists", f"{name}.json")
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False

    def save_screening_result(self, name: str, data: dict):
        """Save screening result to JSON file."""
        file_path = os.path.join(self.base_path, "screening_results", f"{name}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, cls=StockscreenJSONEncoder)

    def load_screening_result(self, name: str) -> Optional[dict]:
        """Load screening result from JSON file."""
        file_path = os.path.join(self.base_path, "screening_results", f"{name}.json")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None


class DefaultSymbols:
    """Manages default symbol lists and market categories."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.cache_file = os.path.join(base_path, "market_data", "default_symbols.json")
        self.cache_expiry = 24 * 60 * 60  # 24 hours

    async def get_symbols(self, category: str | None = None) -> list[str]:
        """Get default symbols, optionally filtered by category.

        Categories: mega_cap, large_cap, mid_cap, small_cap, micro_cap, etf
        """
        symbols_data = await self._load_or_fetch_symbols()

        if category:
            category = category.lower()
            if category not in self._get_category_filters():
                raise ValidationError(f"Invalid category: {category}")
            return self._filter_by_category(symbols_data, category)

        return [s["symbol"] for s in symbols_data]

    def _get_category_filters(self) -> dict:
        """Define market cap and other category filters."""
        return {
            "mega_cap": {"min_cap": 200e9},
            "large_cap": {"min_cap": 10e9, "max_cap": 200e9},
            "mid_cap": {"min_cap": 2e9, "max_cap": 10e9},
            "small_cap": {"min_cap": 300e6, "max_cap": 2e9},
            "micro_cap": {"max_cap": 300e6},
            "etf": {"type": "etf"},
        }

    def _filter_by_category(self, symbols_data: list[dict], category: str) -> list[str]:
        """Filter symbols by category criteria."""
        filters = self._get_category_filters()[category]
        filtered = []

        for data in symbols_data:
            matches = True
            if "type" in filters:
                if data.get("type") != filters["type"]:
                    matches = False
            if "min_cap" in filters and data.get("market_cap"):
                if data["market_cap"] < filters["min_cap"]:
                    matches = False
            if "max_cap" in filters and data.get("market_cap"):
                if data["market_cap"] > filters["max_cap"]:
                    matches = False
            if matches:
                filtered.append(data["symbol"])

        return filtered

    async def _load_or_fetch_symbols(self) -> list[dict]:
        """Load symbols from cache or fetch if expired."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cached = json.load(f)
                if time.time() - cached["timestamp"] < self.cache_expiry:
                    return cached["data"]
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        # Fetch fresh data
        data = await self._fetch_symbols()

        # Save to cache
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump({"timestamp": time.time(), "data": data}, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

        return data

    async def _fetch_symbols(self) -> list[dict]:
        """Fetch symbols from major exchanges.

        Note: This will be delegated to YahooProvider in Phase 3.
        For now, kept as a placeholder that returns an empty list.
        """
        # Will be wired to YahooProvider later
        logger.warning("_fetch_symbols called but provider not yet wired")
        return []
