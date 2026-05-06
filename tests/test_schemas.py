"""Tests for stockscreen.models.schemas module."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from stockscreen.models.schemas import (
    StockscreenJSONEncoder,
    WatchlistName,
    StockSymbols,
)


# ============================================================
# 1. WatchlistName validation
# ============================================================
class TestWatchlistName:
    def test_valid_name(self):
        w = WatchlistName(name="my_watchlist")
        assert w.name == "my_watchlist"

    def test_valid_name_with_numbers_and_hyphen(self):
        w = WatchlistName(name="watch123-test")
        assert w.name == "watch123-test"

    def test_single_char(self):
        w = WatchlistName(name="a")
        assert w.name == "a"

    def test_max_length_50(self):
        w = WatchlistName(name="a" * 50)
        assert len(w.name) == 50

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name="")

    def test_too_long_name_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name="a" * 51)

    def test_starts_with_hyphen_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name="-bad")

    def test_invalid_characters_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name="my watchlist!")

    def test_special_chars_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name="test@name")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError):
            WatchlistName(name=123)


# ============================================================
# 2. StockSymbols validation
# ============================================================
class TestStockSymbols:
    def test_valid_symbols(self):
        s = StockSymbols(symbols=["AAPL", "MSFT", "BRK-B", "BF.B"])
        assert s.symbols == ["AAPL", "MSFT", "BRK-B", "BF.B"]

    def test_empty_list(self):
        s = StockSymbols(symbols=[])
        assert s.symbols == []

    def test_exceeds_max_symbols(self):
        with pytest.raises(ValueError):
            StockSymbols(symbols=["A"] * 1001)

    def test_custom_max_via_validator(self):
        # Default max is 1000, so 999 should pass
        s = StockSymbols(symbols=["A"] * 999)
        assert len(s.symbols) == 999

    def test_symbol_too_long(self):
        with pytest.raises(ValueError):
            StockSymbols(symbols=["TOOLONGSYMBL"])

    def test_symbol_too_short(self):
        with pytest.raises(ValueError):
            StockSymbols(symbols=[""])

    def test_invalid_characters(self):
        with pytest.raises(ValueError):
            StockSymbols(symbols=["AA$L"])

    def test_non_string_symbol(self):
        with pytest.raises(ValueError):
            StockSymbols(symbols=[123])

    def test_symbols_uppercased(self):
        s = StockSymbols(symbols=["aapl", "msft"])
        assert s.symbols == ["AAPL", "MSFT"]


# ============================================================
# 3. StockscreenJSONEncoder
# ============================================================
class TestStockscreenJSONEncoder:
    def _encode(self, obj):
        return json.loads(json.dumps(obj, cls=StockscreenJSONEncoder))

    def test_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        result = self._encode({"ts": ts})
        assert result["ts"] == "2024-01-15T00:00:00"

    def test_nat(self):
        result = self._encode({"val": pd.NaT})
        assert result["val"] == "NaT"

    def test_period(self):
        p = pd.Period("2024-01", freq="M")
        result = self._encode({"p": p})
        assert result["p"] == "2024-01"

    def test_datetime_date(self):
        d = datetime.date(2024, 1, 15)
        result = self._encode({"d": d})
        assert result["d"] == "2024-01-15"

    def test_numpy_int(self):
        result = self._encode({"n": np.int64(42)})
        assert result["n"] == "42"

    def test_numpy_float(self):
        result = self._encode({"n": np.float64(3.14)})
        assert result["n"] == pytest.approx(3.14)

    def test_regular_types_pass_through(self):
        data = {"a": 1, "b": "hello", "c": [1, 2], "d": None}
        assert self._encode(data) == data

    def test_numpy_nan(self):
        result = self._encode({"val": np.nan})
        assert result["val"] is None

    def test_datetime_datetime(self):
        dt = datetime.datetime(2024, 6, 15, 10, 30, 0)
        result = self._encode({"dt": dt})
        assert result["dt"] == "2024-06-15T10:30:00"
