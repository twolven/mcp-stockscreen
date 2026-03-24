"""Tests for the BaseSymbolFetcher interface."""

import pytest

from stockscreen.providers.symbol_fetchers.base import BaseSymbolFetcher, SymbolRecord


# ---------------------------------------------------------------------------
# Concrete stub used in all tests
# ---------------------------------------------------------------------------

class _StubFetcher(BaseSymbolFetcher):
    name = "stub"
    source_url = "https://example.com/stub"

    def __init__(self, records: list[dict] | None = None):
        self._records = records or []

    async def fetch(self) -> list[SymbolRecord]:
        return [SymbolRecord(**r) for r in self._records]


class _MissingNameFetcher(BaseSymbolFetcher):
    source_url = "https://example.com"

    async def fetch(self) -> list[SymbolRecord]:
        return []


class _MissingUrlFetcher(BaseSymbolFetcher):
    name = "no_url"

    async def fetch(self) -> list[SymbolRecord]:
        return []


class _MissingFetchFetcher(BaseSymbolFetcher):
    name = "no_fetch"
    source_url = "https://example.com"


# ---------------------------------------------------------------------------
# 1. ABC enforcement
# ---------------------------------------------------------------------------

class TestBaseSymbolFetcherABC:
    def test_cannot_instantiate_directly(self):
        """BaseSymbolFetcher is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSymbolFetcher()

    def test_subclass_without_fetch_cannot_instantiate(self):
        """Subclass missing fetch() cannot be instantiated."""
        with pytest.raises(TypeError):
            _MissingFetchFetcher()

    def test_subclass_without_name_cannot_instantiate(self):
        """Subclass missing class attribute `name` cannot be instantiated."""
        with pytest.raises(TypeError):
            _MissingNameFetcher()

    def test_subclass_without_source_url_cannot_instantiate(self):
        """Subclass missing class attribute `source_url` cannot be instantiated."""
        with pytest.raises(TypeError):
            _MissingUrlFetcher()

    def test_valid_subclass_instantiates(self):
        """A fully compliant subclass can be instantiated."""
        fetcher = _StubFetcher()
        assert fetcher is not None


# ---------------------------------------------------------------------------
# 2. SymbolRecord dataclass
# ---------------------------------------------------------------------------

class TestSymbolRecord:
    def test_required_fields(self):
        """SymbolRecord requires symbol and name."""
        r = SymbolRecord(symbol="AAPL", name="Apple Inc.")
        assert r.symbol == "AAPL"
        assert r.name == "Apple Inc."

    def test_optional_fields_default_to_none(self):
        """market_cap and instrument_type default to None."""
        r = SymbolRecord(symbol="AAPL", name="Apple Inc.")
        assert r.market_cap is None
        assert r.instrument_type is None

    def test_symbol_uppercased(self):
        """symbol is normalised to uppercase."""
        r = SymbolRecord(symbol="aapl", name="Apple Inc.")
        assert r.symbol == "AAPL"

    def test_symbol_stripped(self):
        """symbol is stripped of whitespace."""
        r = SymbolRecord(symbol="  AAPL  ", name="Apple Inc.")
        assert r.symbol == "AAPL"

    def test_empty_symbol_raises(self):
        """An empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol"):
            SymbolRecord(symbol="", name="Apple Inc.")

    def test_market_cap_must_be_positive(self):
        """market_cap must be a positive number if provided."""
        with pytest.raises(ValueError, match="market_cap"):
            SymbolRecord(symbol="AAPL", name="Apple", market_cap=-1)

    def test_instrument_type_values(self):
        """instrument_type accepts 'equity' and 'etf'."""
        eq = SymbolRecord(symbol="AAPL", name="Apple", instrument_type="equity")
        etf = SymbolRecord(symbol="SPY", name="SPDR", instrument_type="etf")
        assert eq.instrument_type == "equity"
        assert etf.instrument_type == "etf"

    def test_invalid_instrument_type_raises(self):
        """An unknown instrument_type raises ValueError."""
        with pytest.raises(ValueError, match="instrument_type"):
            SymbolRecord(symbol="X", name="X", instrument_type="future")

    def test_to_dict(self):
        """to_dict() returns a plain dict with all fields."""
        r = SymbolRecord(symbol="AAPL", name="Apple Inc.", market_cap=3e12, instrument_type="equity")
        d = r.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["name"] == "Apple Inc."
        assert d["market_cap"] == 3e12
        assert d["instrument_type"] == "equity"


# ---------------------------------------------------------------------------
# 3. fetch() contract
# ---------------------------------------------------------------------------

class TestFetchContract:
    async def test_fetch_returns_list(self):
        """fetch() returns a list."""
        fetcher = _StubFetcher()
        result = await fetcher.fetch()
        assert isinstance(result, list)

    async def test_fetch_returns_symbol_records(self):
        """Each item in the result is a SymbolRecord."""
        fetcher = _StubFetcher([{"symbol": "AAPL", "name": "Apple"}])
        result = await fetcher.fetch()
        assert all(isinstance(r, SymbolRecord) for r in result)

    async def test_fetch_empty_list(self):
        """fetch() may return an empty list (valid)."""
        fetcher = _StubFetcher([])
        result = await fetcher.fetch()
        assert result == []


# ---------------------------------------------------------------------------
# 4. Class-level metadata
# ---------------------------------------------------------------------------

class TestFetcherMetadata:
    def test_name_attribute(self):
        """Fetcher exposes a non-empty `name` string."""
        fetcher = _StubFetcher()
        assert isinstance(fetcher.name, str)
        assert len(fetcher.name) > 0

    def test_source_url_attribute(self):
        """Fetcher exposes a non-empty `source_url` string."""
        fetcher = _StubFetcher()
        assert isinstance(fetcher.source_url, str)
        assert fetcher.source_url.startswith("http")
