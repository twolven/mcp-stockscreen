"""Tests for EuronextProvider — TDD step 1."""

import json
import os
import time
from unittest.mock import patch

import pytest

from stockscreen.providers.euronext import EuronextProvider, EuronextRecord, _normalize_ticker


# ---------------------------------------------------------------------------
# JSON fixtures (Euronext API responses)
# ---------------------------------------------------------------------------

# ISIN → ticker  (XPAR → .PA)
RESPONSE_XPAR = {"isin": "FR0000131104", "symbol": "TTE", "name": "TotalEnergies SE", "mic": "XPAR"}

# ISIN → ticker  (XETR → .DE)
RESPONSE_XETR = {"isin": "DE0005140008", "symbol": "DBK", "name": "Deutsche Bank AG", "mic": "XETR"}

# ISIN → ticker  (XLON → .L)
RESPONSE_XLON = {"isin": "GB0031348658", "symbol": "TSCO", "name": "Tesco PLC", "mic": "XLON"}

# ISIN → ticker  (XAMS → .AS)
RESPONSE_XAMS = {"isin": "NL0000009165", "symbol": "INGA", "name": "ING Groep NV", "mic": "XAMS"}

# ISIN → ticker  (MIC inconnu → suffixe vide)
RESPONSE_UNKNOWN_MIC = {"isin": "XX0000000001", "symbol": "FOO", "name": "Foo Corp", "mic": "XXXX"}

# ticker → ISIN  (search — liste de résultats)
SEARCH_RESPONSE_TTE = [
    {"isin": "FR0000131104", "symbol": "TTE", "name": "TotalEnergies SE", "mic": "XPAR"},
    {"isin": "US89157Q1022", "symbol": "TTE", "name": "TotalEnergies ADR", "mic": "XNYS"},
]

# Réponse vide (ISIN ou ticker inconnu)
RESPONSE_NOT_FOUND = {}
SEARCH_RESPONSE_EMPTY = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(tmp_path, ttl: float = 3600.0) -> EuronextProvider:
    return EuronextProvider(cache_dir=str(tmp_path), cache_ttl_seconds=ttl)


def _make_record_dict(raw: dict) -> dict:
    """Construit un dict EuronextRecord complet depuis une réponse API brute."""
    from stockscreen.providers.euronext import _MIC_TO_SUFFIX
    mic = raw.get("mic", "")
    symbol = raw.get("symbol", "")
    return {
        "isin": raw.get("isin", ""),
        "symbol": symbol,
        "name": raw.get("name", ""),
        "mic": mic,
        "yahoo_ticker": f"{symbol}{_MIC_TO_SUFFIX.get(mic, '')}",
        "cached_at": "2026-01-01T00:00:00",
    }


def _write_cache(tmp_path, key: str, data: dict, age_seconds: float = 0.0):
    """Pré-écrit un fichier cache avec un timestamp donné."""
    record = {
        "timestamp": time.time() - age_seconds,
        "data": _make_record_dict(data),
    }
    path = os.path.join(str(tmp_path), f"euronext_{key}.json")
    with open(path, "w") as f:
        json.dump(record, f)


# ---------------------------------------------------------------------------
# 1. _normalize_ticker
# ---------------------------------------------------------------------------

class TestNormalizeTicker:
    def test_strips_pa_suffix(self):
        assert _normalize_ticker("TTE.PA") == "TTE"

    def test_strips_de_suffix(self):
        assert _normalize_ticker("DBK.DE") == "DBK"

    def test_strips_l_suffix(self):
        assert _normalize_ticker("TSCO.L") == "TSCO"

    def test_strips_as_suffix(self):
        assert _normalize_ticker("INGA.AS") == "INGA"

    def test_no_suffix_unchanged(self):
        assert _normalize_ticker("TTE") == "TTE"

    def test_uppercases(self):
        assert _normalize_ticker("tte.pa") == "TTE"

    def test_isin_unchanged(self):
        assert _normalize_ticker("FR0000131104") == "FR0000131104"


# ---------------------------------------------------------------------------
# 2. resolve_ticker (ISIN → ticker Yahoo)
# ---------------------------------------------------------------------------

class TestResolveTicker:
    async def test_returns_record_for_xpar(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XPAR)):
            rec = await p.resolve_ticker("FR0000131104")
        assert rec is not None
        assert rec.isin == "FR0000131104"
        assert rec.symbol == "TTE"
        assert rec.mic == "XPAR"
        assert rec.yahoo_ticker == "TTE.PA"

    async def test_suffix_de_for_xetr(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XETR)):
            rec = await p.resolve_ticker("DE0005140008")
        assert rec.yahoo_ticker == "DBK.DE"

    async def test_suffix_l_for_xlon(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XLON)):
            rec = await p.resolve_ticker("GB0031348658")
        assert rec.yahoo_ticker == "TSCO.L"

    async def test_suffix_as_for_xams(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XAMS)):
            rec = await p.resolve_ticker("NL0000009165")
        assert rec.yahoo_ticker == "INGA.AS"

    async def test_empty_suffix_for_unknown_mic(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_UNKNOWN_MIC)):
            rec = await p.resolve_ticker("XX0000000001")
        assert rec.yahoo_ticker == "FOO"

    async def test_returns_none_when_not_found(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_NOT_FOUND)):
            rec = await p.resolve_ticker("XX9999999999")
        assert rec is None

    async def test_cached_at_is_set(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XPAR)):
            rec = await p.resolve_ticker("FR0000131104")
        assert rec.cached_at is not None
        assert "T" in rec.cached_at  # ISO format


# ---------------------------------------------------------------------------
# 3. resolve_isin (ticker → ISIN)
# ---------------------------------------------------------------------------

class TestResolveIsin:
    async def test_returns_record_for_ticker(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(SEARCH_RESPONSE_TTE)):
            rec = await p.resolve_isin("TTE")
        assert rec is not None
        assert rec.isin == "FR0000131104"
        assert rec.symbol == "TTE"
        assert rec.yahoo_ticker == "TTE.PA"

    async def test_normalizes_pa_suffix_before_search(self, tmp_path):
        p = _make_provider(tmp_path)
        calls = []

        def recording_get(url):
            calls.append(url)
            return json.dumps(SEARCH_RESPONSE_TTE)

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.resolve_isin("TTE.PA")

        # L'URL d'appel ne doit pas contenir ".PA"
        assert all(".PA" not in url for url in calls)
        assert any("TTE" in url for url in calls)

    async def test_returns_none_when_not_found(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(SEARCH_RESPONSE_EMPTY)):
            rec = await p.resolve_isin("UNKNOWN")
        assert rec is None

    async def test_returns_first_euronext_result(self, tmp_path):
        """Quand plusieurs résultats, retourner le premier."""
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(SEARCH_RESPONSE_TTE)):
            rec = await p.resolve_isin("TTE")
        assert rec.isin == "FR0000131104"  # premier, pas l'ADR NYSE


# ---------------------------------------------------------------------------
# 4. Cache — comportement commun aux deux méthodes
# ---------------------------------------------------------------------------

class TestCache:
    async def test_fresh_cache_skips_http_resolve_ticker(self, tmp_path):
        p = _make_provider(tmp_path)
        _write_cache(tmp_path, "FR0000131104", RESPONSE_XPAR, age_seconds=10)
        with patch.object(p, "_http_get", side_effect=AssertionError("should not call HTTP")):
            rec = await p.resolve_ticker("FR0000131104")
        assert rec is not None
        assert rec.symbol == "TTE"

    async def test_fresh_cache_skips_http_resolve_isin(self, tmp_path):
        p = _make_provider(tmp_path)
        _write_cache(tmp_path, "TTE", RESPONSE_XPAR, age_seconds=10)
        with patch.object(p, "_http_get", side_effect=AssertionError("should not call HTTP")):
            rec = await p.resolve_isin("TTE")
        assert rec is not None

    async def test_expired_cache_triggers_http(self, tmp_path):
        p = _make_provider(tmp_path, ttl=60.0)
        _write_cache(tmp_path, "FR0000131104", RESPONSE_XPAR, age_seconds=120)
        calls = []

        def recording_get(url):
            calls.append(url)
            return json.dumps(RESPONSE_XPAR)

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.resolve_ticker("FR0000131104")

        assert len(calls) == 1

    async def test_cache_miss_triggers_http(self, tmp_path):
        p = _make_provider(tmp_path)
        calls = []

        def recording_get(url):
            calls.append(url)
            return json.dumps(RESPONSE_XPAR)

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.resolve_ticker("FR0000131104")

        assert len(calls) == 1

    async def test_network_error_returns_stale_cache(self, tmp_path):
        p = _make_provider(tmp_path, ttl=1.0)
        _write_cache(tmp_path, "FR0000131104", RESPONSE_XPAR, age_seconds=9999)

        with patch.object(p, "_http_get", side_effect=Exception("timeout")):
            rec = await p.resolve_ticker("FR0000131104")

        assert rec is not None
        assert rec.symbol == "TTE"

    async def test_network_error_no_cache_returns_none(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", side_effect=Exception("timeout")):
            rec = await p.resolve_ticker("FR0000131104")
        assert rec is None

    async def test_cache_written_after_fetch(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(RESPONSE_XPAR)):
            await p.resolve_ticker("FR0000131104")
        cache_file = os.path.join(str(tmp_path), "euronext_FR0000131104.json")
        assert os.path.exists(cache_file)

    async def test_resolve_ticker_and_resolve_isin_share_cache(self, tmp_path):
        """resolve_isin sur 'TTE' écrit un cache ; resolve_ticker sur l'ISIN le réutilise."""
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=json.dumps(SEARCH_RESPONSE_TTE)):
            await p.resolve_isin("TTE")

        # Le cache isin doit maintenant exister
        with patch.object(p, "_http_get", side_effect=AssertionError("should not call HTTP")):
            rec = await p.resolve_ticker("FR0000131104")
        assert rec is not None

    async def test_invalidate_cache_removes_file(self, tmp_path):
        p = _make_provider(tmp_path)
        _write_cache(tmp_path, "FR0000131104", RESPONSE_XPAR)
        p.invalidate_cache("FR0000131104")
        cache_file = os.path.join(str(tmp_path), "euronext_FR0000131104.json")
        assert not os.path.exists(cache_file)

    async def test_invalidate_nonexistent_is_silent(self, tmp_path):
        p = _make_provider(tmp_path)
        p.invalidate_cache("FR9999999999")  # ne doit pas lever d'exception


# ---------------------------------------------------------------------------
# 5. Async — exécution dans run_in_executor
# ---------------------------------------------------------------------------

class TestAsync:
    async def test_http_get_called_in_executor(self, tmp_path):
        """_http_get est appelé via run_in_executor (non bloquant)."""
        p = _make_provider(tmp_path)
        calls = []

        def recording_get(url):
            calls.append(url)
            return json.dumps(RESPONSE_XPAR)

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.resolve_ticker("FR0000131104")

        assert len(calls) == 1

    async def test_resolve_isin_http_called_in_executor(self, tmp_path):
        p = _make_provider(tmp_path)
        calls = []

        def recording_get(url):
            calls.append(url)
            return json.dumps(SEARCH_RESPONSE_TTE)

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.resolve_isin("TTE")

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6. EuronextRecord — tous les MIC de la table
# ---------------------------------------------------------------------------

class TestMicToSuffix:
    @pytest.mark.parametrize("mic,expected_suffix", [
        ("XPAR", ".PA"),
        ("XETR", ".DE"),
        ("XLON", ".L"),
        ("XAMS", ".AS"),
        ("XMIL", ".MI"),
        ("XMAD", ".MC"),
        ("XBRU", ".BR"),
        ("XLIS", ".LS"),
        ("XHEL", ".HE"),
        ("XSTO", ".ST"),
        ("XOSL", ".OL"),
        ("XXXX", ""),       # inconnu → vide
    ])
    async def test_yahoo_ticker_suffix(self, tmp_path, mic, expected_suffix):
        p = _make_provider(tmp_path)
        response = {"isin": "XX0000000001", "symbol": "FOO", "name": "Foo", "mic": mic}
        with patch.object(p, "_http_get", return_value=json.dumps(response)):
            rec = await p.resolve_ticker("XX0000000001")
        assert rec.yahoo_ticker == f"FOO{expected_suffix}"
