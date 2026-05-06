"""Tests for PalmaresStore — TDD step 5."""

import json
import os

import pytest

from stockscreen.providers.boursorama_palmares import PalmaresEntry
from stockscreen.store.palmares_store import PalmaresSnapshot, PalmaresStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entries() -> list[PalmaresEntry]:
    return [
        PalmaresEntry(
            code_bourso="1rPMMT",
            nom="M6 METROPOLE TELE.",
            cours=11.36,
            dividendes=[
                {"annee": "2025", "dividende": 1.25, "rendement": 10.70},
                {"annee": "2026", "dividende": 1.25, "rendement": 10.70},
            ],
            isin="FR0000053225",
        ),
        PalmaresEntry(
            code_bourso="1rPICAD",
            nom="ICADE",
            cours=19.14,
            dividendes=[
                {"annee": "2025", "dividende": 1.92, "rendement": 9.79},
            ],
            isin=None,
        ),
    ]


def _make_snapshot(entries=None) -> PalmaresSnapshot:
    return PalmaresSnapshot(
        fetched_at="2026-03-24T10:00:00",
        page_count=3,
        total_entries=2,
        entries=entries or _make_entries(),
    )


def _make_store(tmp_path) -> PalmaresStore:
    return PalmaresStore(base_path=str(tmp_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPalmaresStore:
    def test_load_returns_none_when_file_absent(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load() is None

    def test_save_creates_file(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot())
        assert os.path.exists(store._path())

    def test_save_creates_directory(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot())
        assert os.path.isdir(os.path.join(str(tmp_path), "palmares"))

    def test_round_trip_preserves_all_fields(self, tmp_path):
        store = _make_store(tmp_path)
        snap = _make_snapshot()
        store.save(snap)
        loaded = store.load()
        assert loaded is not None
        assert loaded.fetched_at == "2026-03-24T10:00:00"
        assert loaded.page_count == 3
        assert loaded.total_entries == 2

    def test_round_trip_preserves_entries(self, tmp_path):
        store = _make_store(tmp_path)
        snap = _make_snapshot()
        store.save(snap)
        loaded = store.load()
        assert len(loaded.entries) == 2
        e = loaded.entries[0]
        assert e.code_bourso == "1rPMMT"
        assert e.nom == "M6 METROPOLE TELE."
        assert e.cours == pytest.approx(11.36)
        assert e.isin == "FR0000053225"

    def test_round_trip_preserves_dividendes_list(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot())
        loaded = store.load()
        divs = loaded.entries[0].dividendes
        assert len(divs) == 2
        assert divs[0]["annee"] == "2025"
        assert divs[0]["dividende"] == pytest.approx(1.25)
        assert divs[0]["rendement"] == pytest.approx(10.70)

    def test_round_trip_preserves_none_isin(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot())
        loaded = store.load()
        assert loaded.entries[1].isin is None

    def test_save_overwrites_previous_snapshot(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot(_make_entries()[:1]))
        store.save(_make_snapshot(_make_entries()))
        loaded = store.load()
        assert len(loaded.entries) == 2

    def test_json_file_is_human_readable(self, tmp_path):
        store = _make_store(tmp_path)
        store.save(_make_snapshot())
        with open(store._path()) as f:
            content = f.read()
        # Indented JSON has newlines
        assert "\n" in content

    def test_corrupted_file_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        os.makedirs(os.path.dirname(store._path()), exist_ok=True)
        with open(store._path(), "w") as f:
            f.write("{ not valid json }")
        assert store.load() is None
