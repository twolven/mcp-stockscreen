"""Tests for stockscreen.store.data_store module."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from stockscreen.exceptions import ValidationError
from stockscreen.store.data_store import DefaultSymbols, ScreenerDataStore


# ============================================================
# 1. ScreenerDataStore — directories
# ============================================================
class TestScreenerDataStoreDirectories:
    def test_ensure_directories_created(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        assert (tmp_path / "screening_results").is_dir()
        assert (tmp_path / "watchlists").is_dir()
        assert (tmp_path / "market_data").is_dir()

    def test_default_symbols_initialized(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        assert isinstance(store.default_symbols, DefaultSymbols)


# ============================================================
# 2. ScreenerDataStore — watchlists CRUD
# ============================================================
class TestScreenerDataStoreWatchlists:
    def test_save_and_load_watchlist(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        store.save_watchlist("test", ["AAPL", "MSFT"])
        result = store.load_watchlist("test")
        assert result == ["AAPL", "MSFT"]

    def test_load_nonexistent_watchlist(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        assert store.load_watchlist("nonexistent") is None

    def test_delete_watchlist(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        store.save_watchlist("todelete", ["AAPL"])
        assert store.delete_watchlist("todelete") is True
        assert store.load_watchlist("todelete") is None

    def test_delete_nonexistent_watchlist(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        assert store.delete_watchlist("nonexistent") is False

    def test_save_watchlist_overwrites(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        store.save_watchlist("overwrite", ["AAPL"])
        store.save_watchlist("overwrite", ["MSFT", "GOOG"])
        assert store.load_watchlist("overwrite") == ["MSFT", "GOOG"]


# ============================================================
# 3. ScreenerDataStore — screening results CRUD
# ============================================================
class TestScreenerDataStoreResults:
    def test_save_and_load_screening_result(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        data = {"screen_type": "technical", "results": [{"symbol": "AAPL"}]}
        store.save_screening_result("test_result", data)
        result = store.load_screening_result("test_result")
        assert result == data

    def test_load_nonexistent_screening_result(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        assert store.load_screening_result("nonexistent") is None

    def test_screening_result_with_special_types(self, tmp_path):
        store = ScreenerDataStore(base_path=str(tmp_path))
        data = {"timestamp": pd.Timestamp("2024-01-01"), "value": pd.NaT}
        store.save_screening_result("special", data)
        result = store.load_screening_result("special")
        assert result["timestamp"] == "2024-01-01T00:00:00"
        assert result["value"] == "NaT"


# ============================================================
# 4. DefaultSymbols — category filtering
# ============================================================
class TestDefaultSymbolsFiltering:
    def test_filter_mega_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "AAPL", "market_cap": 3_000_000_000_000, "type": "equity"},
            {"symbol": "SMLL", "market_cap": 500_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "mega_cap")
        assert result == ["AAPL"]

    def test_filter_large_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "LRG", "market_cap": 50_000_000_000, "type": "equity"},
            {"symbol": "MEGA", "market_cap": 300_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "large_cap")
        assert result == ["LRG"]

    def test_filter_mid_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "MID", "market_cap": 5_000_000_000, "type": "equity"},
            {"symbol": "BIG", "market_cap": 50_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "mid_cap")
        assert result == ["MID"]

    def test_filter_small_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "SMLL", "market_cap": 1_000_000_000, "type": "equity"},
            {"symbol": "BIG", "market_cap": 50_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "small_cap")
        assert result == ["SMLL"]

    def test_filter_micro_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "MICRO", "market_cap": 100_000_000, "type": "equity"},
            {"symbol": "BIG", "market_cap": 50_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "micro_cap")
        assert result == ["MICRO"]

    def test_filter_etf(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "SPY", "market_cap": None, "type": "etf"},
            {"symbol": "AAPL", "market_cap": 3_000_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "etf")
        assert result == ["SPY"]


# ============================================================
# 5. DefaultSymbols — async get_symbols
# ============================================================
class TestDefaultSymbolsAsync:
    async def test_get_symbols_invalid_category(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        with patch.object(ds, "_load_or_fetch_symbols", new_callable=AsyncMock, return_value=[]):
            with pytest.raises(ValidationError, match="Invalid category"):
                await ds.get_symbols(category="invalid_category")

    async def test_get_symbols_returns_all_when_no_category(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "AAPL", "market_cap": 3e12, "type": "equity"},
            {"symbol": "SPY", "market_cap": None, "type": "etf"},
        ]
        with patch.object(ds, "_load_or_fetch_symbols", new_callable=AsyncMock, return_value=symbols_data):
            result = await ds.get_symbols()
            assert result == ["AAPL", "SPY"]

    async def test_get_symbols_filtered_by_category(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "AAPL", "market_cap": 3e12, "type": "equity"},
            {"symbol": "SMLL", "market_cap": 500e6, "type": "equity"},
        ]
        with patch.object(ds, "_load_or_fetch_symbols", new_callable=AsyncMock, return_value=symbols_data):
            result = await ds.get_symbols(category="mega_cap")
            assert result == ["AAPL"]


# ============================================================
# 6. DefaultSymbols — cache
# ============================================================
class TestDefaultSymbolsCache:
    async def test_load_from_cache(self, tmp_path):
        ds = DefaultSymbols(base_path=str(tmp_path))
        cache_dir = tmp_path / "market_data"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cached_data = {
            "timestamp": time.time(),
            "data": [{"symbol": "CACHED", "market_cap": 1e12, "type": "equity"}],
        }
        with open(ds.cache_file, "w") as f:
            json.dump(cached_data, f)

        result = await ds._load_or_fetch_symbols()
        assert result == [{"symbol": "CACHED", "market_cap": 1e12, "type": "equity"}]

    async def test_expired_cache_triggers_fetch(self, tmp_path):
        ds = DefaultSymbols(base_path=str(tmp_path))
        cache_dir = tmp_path / "market_data"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache expired (timestamp 48h ago)
        cached_data = {
            "timestamp": time.time() - 48 * 3600,
            "data": [{"symbol": "OLD", "market_cap": 1e12, "type": "equity"}],
        }
        with open(ds.cache_file, "w") as f:
            json.dump(cached_data, f)

        fresh_data = [{"symbol": "FRESH", "market_cap": 2e12, "type": "equity"}]
        with patch.object(ds, "_fetch_symbols", new_callable=AsyncMock, return_value=fresh_data):
            result = await ds._load_or_fetch_symbols()
            assert result == fresh_data
