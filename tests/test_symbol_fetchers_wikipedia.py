"""Tests for Wikipedia-based symbol fetchers."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from stockscreen.exceptions import APIError
from stockscreen.providers.symbol_fetchers.base import SymbolRecord
from stockscreen.providers.symbol_fetchers.wikipedia import (
    SP500Fetcher,
    Nasdaq100Fetcher,
    CAC40Fetcher,
    SBF120Fetcher,
    DAXFetcher,
    FTSE100Fetcher,
    AEXFetcher,
)

ALL_FETCHERS = [
    SP500Fetcher,
    Nasdaq100Fetcher,
    CAC40Fetcher,
    SBF120Fetcher,
    DAXFetcher,
    FTSE100Fetcher,
    AEXFetcher,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(symbol_col: str, name_col: str, rows: list[tuple]) -> list[pd.DataFrame]:
    """Build a fake pd.read_html return value."""
    df = pd.DataFrame(rows, columns=[symbol_col, name_col])
    return [df]


def _patch_read_html(fetcher_instance, tables: list[pd.DataFrame]):
    """Context manager that patches pd.read_html for the fetcher's module."""
    return patch(
        "stockscreen.providers.symbol_fetchers.wikipedia.pd.read_html",
        return_value=tables,
    )


# ---------------------------------------------------------------------------
# 1. Interface compliance
# ---------------------------------------------------------------------------

class TestFetcherInterface:
    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_has_name(self, cls):
        assert isinstance(cls.name, str) and cls.name

    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_has_source_url(self, cls):
        assert cls.source_url.startswith("https://en.wikipedia.org")

    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_name_is_unique(self, cls):
        names = [c.name for c in ALL_FETCHERS]
        assert names.count(cls.name) == 1

    @pytest.mark.parametrize("cls", ALL_FETCHERS)
    def test_instantiates(self, cls):
        assert cls() is not None


# ---------------------------------------------------------------------------
# 2. SP500Fetcher
# ---------------------------------------------------------------------------

class TestSP500Fetcher:
    async def test_fetch_returns_symbol_records(self):
        fetcher = SP500Fetcher()
        tables = _make_table("Symbol", "Security", [("AAPL", "Apple Inc."), ("MSFT", "Microsoft")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert all(isinstance(r, SymbolRecord) for r in result)
        assert len(result) == 2

    async def test_symbols_have_no_suffix(self):
        fetcher = SP500Fetcher()
        tables = _make_table("Symbol", "Security", [("AAPL", "Apple Inc.")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "AAPL"

    async def test_names_extracted(self):
        fetcher = SP500Fetcher()
        tables = _make_table("Symbol", "Security", [("AAPL", "Apple Inc.")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].name == "Apple Inc."

    async def test_instrument_type_is_equity(self):
        fetcher = SP500Fetcher()
        tables = _make_table("Symbol", "Security", [("AAPL", "Apple Inc.")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].instrument_type == "equity"

    async def test_empty_table_returns_empty_list(self):
        fetcher = SP500Fetcher()
        tables = [pd.DataFrame(columns=["Symbol", "Security"])]
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result == []

    async def test_network_error_raises_api_error(self):
        fetcher = SP500Fetcher()
        with patch(
            "stockscreen.providers.symbol_fetchers.wikipedia.pd.read_html",
            side_effect=Exception("timeout"),
        ):
            with pytest.raises(APIError):
                await fetcher.fetch()

    async def test_skips_nan_symbols(self):
        """Rows with NaN / empty ticker are silently skipped."""
        fetcher = SP500Fetcher()
        df = pd.DataFrame({"Symbol": ["AAPL", None, ""], "Security": ["Apple", "Bad", "Also bad"]})
        with _patch_read_html(fetcher, [df]):
            result = await fetcher.fetch()
        assert len(result) == 1
        assert result[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# 3. Nasdaq100Fetcher
# ---------------------------------------------------------------------------

class TestNasdaq100Fetcher:
    async def test_fetch_returns_records(self):
        fetcher = Nasdaq100Fetcher()
        tables = _make_table("Ticker", "Company", [("AAPL", "Apple"), ("NVDA", "Nvidia")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert len(result) == 2
        assert result[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# 4. CAC40Fetcher
# ---------------------------------------------------------------------------

class TestCAC40Fetcher:
    async def test_symbols_get_pa_suffix(self):
        fetcher = CAC40Fetcher()
        tables = _make_table("Ticker symbol", "Company", [("TTE", "TotalEnergies"), ("AI", "Air Liquide")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "TTE.PA"
        assert result[1].symbol == "AI.PA"

    async def test_no_double_suffix(self):
        """Symbols that already carry .PA are not double-suffixed."""
        fetcher = CAC40Fetcher()
        tables = _make_table("Ticker symbol", "Company", [("TTE.PA", "TotalEnergies")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "TTE.PA"

    async def test_network_error_raises_api_error(self):
        fetcher = CAC40Fetcher()
        with patch(
            "stockscreen.providers.symbol_fetchers.wikipedia.pd.read_html",
            side_effect=Exception("503"),
        ):
            with pytest.raises(APIError):
                await fetcher.fetch()


# ---------------------------------------------------------------------------
# 5. SBF120Fetcher
# ---------------------------------------------------------------------------

class TestSBF120Fetcher:
    async def test_symbols_get_pa_suffix(self):
        fetcher = SBF120Fetcher()
        tables = _make_table("Ticker symbol", "Company", [("ORA", "Orange")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "ORA.PA"


# ---------------------------------------------------------------------------
# 6. DAXFetcher
# ---------------------------------------------------------------------------

class TestDAXFetcher:
    async def test_symbols_get_de_suffix(self):
        fetcher = DAXFetcher()
        tables = _make_table("Ticker symbol", "Company", [("ADS", "Adidas"), ("BMW", "BMW")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "ADS.DE"
        assert result[1].symbol == "BMW.DE"


# ---------------------------------------------------------------------------
# 7. FTSE100Fetcher
# ---------------------------------------------------------------------------

class TestFTSE100Fetcher:
    async def test_symbols_get_l_suffix(self):
        fetcher = FTSE100Fetcher()
        tables = _make_table("Ticker", "Company", [("SHEL", "Shell"), ("HSBA", "HSBC")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "SHEL.L"
        assert result[1].symbol == "HSBA.L"


# ---------------------------------------------------------------------------
# 8. AEXFetcher
# ---------------------------------------------------------------------------

class TestAEXFetcher:
    async def test_symbols_get_as_suffix(self):
        fetcher = AEXFetcher()
        tables = _make_table("Ticker symbol", "Company", [("ASML", "ASML"), ("INGA", "ING")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == "ASML.AS"
        assert result[1].symbol == "INGA.AS"


# ---------------------------------------------------------------------------
# 9. Common behaviour across all fetchers
# ---------------------------------------------------------------------------

class TestCommonBehaviour:
    @pytest.mark.parametrize("cls,symbol_col,name_col,raw_symbol,expected", [
        (SP500Fetcher,    "Symbol",       "Security",     "AAPL",  "AAPL"),
        (Nasdaq100Fetcher,"Ticker",       "Company",      "NVDA",  "NVDA"),
        (CAC40Fetcher,    "Ticker symbol","Company",      "TTE",   "TTE.PA"),
        (SBF120Fetcher,   "Ticker symbol","Company",      "ORA",   "ORA.PA"),
        (DAXFetcher,      "Ticker symbol","Company",      "BMW",   "BMW.DE"),
        (FTSE100Fetcher,  "Ticker",       "Company",      "SHEL",  "SHEL.L"),
        (AEXFetcher,      "Ticker symbol","Company",      "ASML",  "ASML.AS"),
    ])
    async def test_suffix_applied_correctly(self, cls, symbol_col, name_col, raw_symbol, expected):
        fetcher = cls()
        tables = _make_table(symbol_col, name_col, [(raw_symbol, "Some Company")])
        with _patch_read_html(fetcher, tables):
            result = await fetcher.fetch()
        assert result[0].symbol == expected
