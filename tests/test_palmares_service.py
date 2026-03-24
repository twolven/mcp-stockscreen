"""Tests for PalmaresService — TDD step 6."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stockscreen.providers.boursorama_palmares import PalmaresEntry
from stockscreen.services.palmares_service import PalmaresService
from stockscreen.store.palmares_store import PalmaresSnapshot, PalmaresStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entry(code="1rXXX", nom="Corp X", cours=50.0, rendement_2025=5.0, isin=None) -> PalmaresEntry:
    return PalmaresEntry(
        code_bourso=code,
        nom=nom,
        cours=cours,
        dividendes=[{"annee": "2025", "dividende": cours * rendement_2025 / 100, "rendement": rendement_2025}],
        isin=isin,
    )


def _snapshot(entries, fetched_at="2026-03-24T10:00:00", page_count=3) -> PalmaresSnapshot:
    return PalmaresSnapshot(
        fetched_at=fetched_at,
        page_count=page_count,
        total_entries=len(entries),
        entries=entries,
    )


def _make_service(tmp_path, ttl=3600.0, entries=None) -> tuple[PalmaresService, AsyncMock]:
    scraper = AsyncMock()
    if entries is None:
        entries = [
            _entry("1rPA", "Alpha", rendement_2025=8.0),
            _entry("1rPB", "Beta",  rendement_2025=5.0),
            _entry("1rPC", "Gamma", rendement_2025=3.0),
        ]
    scraper.fetch_all = AsyncMock(return_value=entries)
    store = MagicMock(spec=PalmaresStore)
    store.load.return_value = None  # cache miss by default
    svc = PalmaresService(scraper=scraper, store=store, cache_ttl_seconds=ttl)
    return svc, scraper


# ---------------------------------------------------------------------------
# 1. Cache logic
# ---------------------------------------------------------------------------

class TestCacheLogic:
    async def test_cache_miss_triggers_scraper(self, tmp_path):
        svc, scraper = _make_service(tmp_path)
        await svc.get()
        scraper.fetch_all.assert_called_once()

    async def test_fresh_cache_skips_scraper(self, tmp_path):
        svc, scraper = _make_service(tmp_path, ttl=3600.0)
        fresh_snap = _snapshot([_entry()])
        svc._store.load.return_value = fresh_snap
        # Marquer comme frais
        with patch.object(svc, "_is_fresh", return_value=True):
            await svc.get()
        scraper.fetch_all.assert_not_called()

    async def test_expired_cache_triggers_scraper(self, tmp_path):
        svc, scraper = _make_service(tmp_path, ttl=1.0)
        old_snap = _snapshot([_entry()])
        svc._store.load.return_value = old_snap
        with patch.object(svc, "_is_fresh", return_value=False):
            await svc.get()
        scraper.fetch_all.assert_called_once()

    async def test_refresh_forces_scraper_even_if_fresh(self, tmp_path):
        svc, scraper = _make_service(tmp_path)
        fresh_snap = _snapshot([_entry()])
        svc._store.load.return_value = fresh_snap
        with patch.object(svc, "_is_fresh", return_value=True):
            await svc.refresh()
        scraper.fetch_all.assert_called_once()

    async def test_snapshot_saved_after_scraping(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        await svc.get()
        svc._store.save.assert_called_once()

    async def test_refresh_returns_snapshot(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        snap = await svc.refresh()
        assert isinstance(snap, PalmaresSnapshot)
        assert snap.total_entries == 3


# ---------------------------------------------------------------------------
# 2. Tri
# ---------------------------------------------------------------------------

class TestSorting:
    async def test_sorted_by_rendement_descending(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get()
        rendements = [
            e.dividendes[0]["rendement"]
            for e in result.entries
            if e.dividendes and e.dividendes[0]["rendement"] is not None
        ]
        assert rendements == sorted(rendements, reverse=True)

    async def test_entries_with_none_rendement_at_end(self, tmp_path):
        entries = [
            _entry("1rPA", rendement_2025=5.0),
            PalmaresEntry("1rPNone", "NoDiv", 10.0, [{"annee": "2025", "dividende": None, "rendement": None}]),
            _entry("1rPB", rendement_2025=8.0),
        ]
        svc, _ = _make_service(tmp_path, entries=entries)
        result = await svc.get()
        # Le None doit être en dernier
        last = result.entries[-1]
        assert last.code_bourso == "1rPNone"


# ---------------------------------------------------------------------------
# 3. Filtres
# ---------------------------------------------------------------------------

class TestFilters:
    async def test_filter_min_rendement(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(min_rendement=5.0)
        for e in result.entries:
            rend = e.dividendes[0]["rendement"] if e.dividendes else None
            assert rend is None or rend >= 5.0

    async def test_filter_max_rendement(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(max_rendement=6.0)
        for e in result.entries:
            rend = e.dividendes[0]["rendement"] if e.dividendes else None
            assert rend is None or rend <= 6.0

    async def test_filter_min_and_max_rendement_combined(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(min_rendement=4.0, max_rendement=7.0)
        for e in result.entries:
            rend = e.dividendes[0]["rendement"] if e.dividendes else None
            assert rend is None or (4.0 <= rend <= 7.0)

    async def test_filter_nom_contains_case_insensitive(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(nom_contains="alpha")
        assert len(result.entries) == 1
        assert result.entries[0].nom == "Alpha"

    async def test_filter_nom_contains_partial_match(self, tmp_path):
        entries = [
            _entry(nom="TotalEnergies SE"),
            _entry(nom="Total Gabon"),
            _entry(nom="Airbus SE"),
        ]
        svc, _ = _make_service(tmp_path, entries=entries)
        result = await svc.get(nom_contains="total")
        assert len(result.entries) == 2

    async def test_filter_limit(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(limit=2)
        assert len(result.entries) <= 2

    async def test_filters_combined(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(min_rendement=5.0, nom_contains="alpha")
        assert all("Alpha" in e.nom for e in result.entries)


# ---------------------------------------------------------------------------
# 4. Métadonnées du snapshot retourné
# ---------------------------------------------------------------------------

class TestSnapshotMetadata:
    async def test_total_entries_reflects_unfiltered_count(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(min_rendement=99.0)  # filtre drastique
        # total_entries = taille avant filtrage
        assert result.total_entries == 3

    async def test_returned_entries_match_filter(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get(min_rendement=99.0)
        assert len(result.entries) == 0

    async def test_fetched_at_is_set(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        result = await svc.get()
        assert result.fetched_at is not None
        assert "T" in result.fetched_at
