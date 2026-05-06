"""Tests for SymbolService — cache, refresh, and background scheduling."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from stockscreen.exceptions import APIError, ValidationError
from stockscreen.providers.symbol_fetchers.base import BaseSymbolFetcher, SymbolRecord
from stockscreen.services.symbol_service import SymbolService


# ---------------------------------------------------------------------------
# Stub fetcher
# ---------------------------------------------------------------------------

class _FakeFetcher(BaseSymbolFetcher):
    name = "fake"
    source_url = "https://example.com/fake"

    def __init__(self, records: list[SymbolRecord] | None = None, raises: Exception | None = None):
        super().__init__()
        self._records = records or [
            SymbolRecord(symbol="AAA", name="Company A"),
            SymbolRecord(symbol="BBB", name="Company B"),
        ]
        self._raises = raises
        self.call_count = 0

    async def fetch(self) -> list[SymbolRecord]:
        self.call_count += 1
        if self._raises:
            raise self._raises
        return self._records


class _AnotherFetcher(BaseSymbolFetcher):
    name = "other"
    source_url = "https://example.com/other"

    async def fetch(self) -> list[SymbolRecord]:
        return [SymbolRecord(symbol="OOO", name="Other Co")]


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

class TestSymbolServiceInit:
    def test_instantiates_with_fetchers(self, tmp_path):
        svc = SymbolService(
            fetchers=[_FakeFetcher()],
            cache_dir=str(tmp_path),
        )
        assert svc is not None

    def test_available_categories_matches_fetcher_names(self, tmp_path):
        svc = SymbolService(
            fetchers=[_FakeFetcher(), _AnotherFetcher()],
            cache_dir=str(tmp_path),
        )
        assert set(svc.available_categories()) == {"fake", "other"}

    def test_empty_fetcher_list_allowed(self, tmp_path):
        svc = SymbolService(fetchers=[], cache_dir=str(tmp_path))
        assert svc.available_categories() == []

    def test_duplicate_fetcher_names_raise_validation_error(self, tmp_path):
        class _DupFetcher1(BaseSymbolFetcher):
            name = "dup"
            source_url = "https://example.com/a"

            async def fetch(self) -> list[SymbolRecord]:
                return []

        class _DupFetcher2(BaseSymbolFetcher):
            name = "dup"
            source_url = "https://example.com/b"

            async def fetch(self) -> list[SymbolRecord]:
                return []

        with pytest.raises(ValidationError, match="Duplicate"):
            SymbolService(
                fetchers=[_DupFetcher1(), _DupFetcher2()],
                cache_dir=str(tmp_path),
            )

    def test_non_duplicate_fetchers_still_work(self, tmp_path):
        svc = SymbolService(
            fetchers=[_FakeFetcher(), _AnotherFetcher()],
            cache_dir=str(tmp_path),
        )
        assert set(svc.available_categories()) == {"fake", "other"}


# ---------------------------------------------------------------------------
# 2. get() — symbol resolution
# ---------------------------------------------------------------------------

class TestSymbolServiceGet:
    async def test_get_calls_fetcher_when_no_cache(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        symbols = await svc.get("fake")

        assert fetcher.call_count == 1
        assert symbols == ["AAA", "BBB"]

    async def test_get_returns_cached_symbols_without_refetch(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path), refresh_interval_hours=24)

        await svc.get("fake")   # populates cache
        await svc.get("fake")   # should hit cache

        assert fetcher.call_count == 1

    async def test_get_refetches_when_cache_expired(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path), refresh_interval_hours=24)

        # Write a stale cache manually
        cache_path = tmp_path / "fake.json"
        cache_path.write_text(json.dumps({
            "timestamp": time.time() - 25 * 3600,   # 25 hours ago
            "symbols": [{"symbol": "OLD", "name": "Old Co", "market_cap": None, "instrument_type": None}],
        }))

        symbols = await svc.get("fake")

        assert fetcher.call_count == 1          # re-fetched
        assert "OLD" not in symbols             # replaced by fresh data
        assert "AAA" in symbols

    async def test_get_raises_validation_error_for_unknown_category(self, tmp_path):
        svc = SymbolService(fetchers=[_FakeFetcher()], cache_dir=str(tmp_path))
        with pytest.raises(ValidationError, match="Unknown"):
            await svc.get("nonexistent")

    async def test_get_persists_cache_to_disk(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        await svc.get("fake")

        cache_file = tmp_path / "fake.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert "timestamp" in data
        assert len(data["symbols"]) == 2

    async def test_get_loads_cache_from_disk_on_new_instance(self, tmp_path):
        """A fresh SymbolService instance picks up the on-disk cache."""
        fetcher1 = _FakeFetcher()
        svc1 = SymbolService(fetchers=[fetcher1], cache_dir=str(tmp_path))
        await svc1.get("fake")

        fetcher2 = _FakeFetcher()
        svc2 = SymbolService(fetchers=[fetcher2], cache_dir=str(tmp_path), refresh_interval_hours=24)
        await svc2.get("fake")

        assert fetcher2.call_count == 0   # served from disk cache

    async def test_get_raises_api_error_when_fetcher_fails_and_no_cache(self, tmp_path):
        fetcher = _FakeFetcher(raises=APIError("network error"))
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        with pytest.raises(APIError):
            await svc.get("fake")

    async def test_get_returns_stale_cache_when_fetcher_fails(self, tmp_path):
        """On fetch failure, fall back to the previous (stale) cache."""
        stale_cache = {
            "timestamp": time.time() - 25 * 3600,
            "symbols": [{"symbol": "STALE", "name": "Stale Co", "market_cap": None, "instrument_type": None}],
        }
        (tmp_path / "fake.json").write_text(json.dumps(stale_cache))

        fetcher = _FakeFetcher(raises=APIError("network error"))
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        symbols = await svc.get("fake")
        assert symbols == ["STALE"]


# ---------------------------------------------------------------------------
# 3. refresh()
# ---------------------------------------------------------------------------

class TestSymbolServiceRefresh:
    async def test_refresh_specific_category_returns_count(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        result = await svc.refresh("fake")

        assert result == {"fake": 2}
        assert fetcher.call_count == 1

    async def test_refresh_all_returns_count_per_category(self, tmp_path):
        svc = SymbolService(
            fetchers=[_FakeFetcher(), _AnotherFetcher()],
            cache_dir=str(tmp_path),
        )

        result = await svc.refresh()

        assert "fake" in result
        assert "other" in result
        assert result["fake"] == 2
        assert result["other"] == 1

    async def test_refresh_force_updates_fresh_cache(self, tmp_path):
        """refresh() always re-fetches, even when cache is fresh."""
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        await svc.get("fake")       # populates fresh cache
        await svc.refresh("fake")   # must still call fetcher

        assert fetcher.call_count == 2

    async def test_refresh_unknown_category_raises_validation_error(self, tmp_path):
        svc = SymbolService(fetchers=[_FakeFetcher()], cache_dir=str(tmp_path))
        with pytest.raises(ValidationError, match="Unknown"):
            await svc.refresh("nonexistent")

    async def test_refresh_records_error_in_result(self, tmp_path):
        """When a fetcher fails, refresh() records the error instead of raising."""
        fetcher = _FakeFetcher(raises=APIError("fail"))
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        result = await svc.refresh()

        assert "error" in result["fake"]


# ---------------------------------------------------------------------------
# 4. Background refresh loop
# ---------------------------------------------------------------------------

class TestBackgroundRefresh:
    async def test_background_refresh_triggers_on_expired_cache(self, tmp_path):
        """_run_background_tick() refreshes categories whose cache is expired."""
        stale = {
            "timestamp": time.time() - 25 * 3600,
            "symbols": [{"symbol": "OLD", "name": "Old", "market_cap": None, "instrument_type": None}],
        }
        (tmp_path / "fake.json").write_text(json.dumps(stale))

        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path), refresh_interval_hours=24)

        await svc._run_background_tick()

        assert fetcher.call_count == 1

    async def test_background_refresh_skips_fresh_cache(self, tmp_path):
        fetcher = _FakeFetcher()
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path), refresh_interval_hours=24)
        await svc.get("fake")  # populate fresh cache

        await svc._run_background_tick()

        assert fetcher.call_count == 1  # only the initial get(), not the tick

    async def test_background_refresh_logs_error_and_continues(self, tmp_path):
        """A failing fetcher in the background tick is logged but doesn't raise."""
        fetcher = _FakeFetcher(raises=APIError("network error"))
        svc = SymbolService(fetchers=[fetcher], cache_dir=str(tmp_path))

        # No exception should propagate
        await svc._run_background_tick()
