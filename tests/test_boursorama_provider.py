"""Tests for BoursoramaProvider."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from stockscreen.exceptions import APIError
from stockscreen.providers.boursorama import (
    BoursoramaProvider,
    BoursoramaQuote,
    _parse_date,
    _parse_float,
    _parse_float_with_currency,
)


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

SEARCH_HTML_EURONEXT = """
<div>
  <a class="search__list-link" href="/cours/1rTTE/">
    <span class="search__item-title">TotalEnergies SE</span>
    <span class="search__item-content">Euronext Paris</span>
  </a>
  <a class="search__list-link" href="/cours/us-TTE/">
    <span class="search__item-title">TotalEnergies ADR</span>
    <span class="search__item-content">NYSE</span>
  </a>
</div>
"""

SEARCH_HTML_NO_RESULT = "<div><p>Aucun résultat</p></div>"

SEARCH_HTML_MULTI_EXCHANGE = """
<div>
  <a class="search__list-link" href="/cours/1rAIR/">
    <span class="search__item-title">Airbus SE</span>
    <span class="search__item-content">Euronext Paris</span>
  </a>
  <a class="search__list-link" href="/cours/xetra-AIR/">
    <span class="search__item-title">Airbus SE</span>
    <span class="search__item-content">XETRA</span>
  </a>
</div>
"""

# Old markup with <h4> (backwards compatibility)
SEARCH_HTML_LEGACY_H4 = """
<div>
  <a class="search__list-link" href="/cours/1rOR/">
    <h4 class="search__item-title">L'Oréal</h4>
    <p class="search__item-content">Euronext Paris</p>
  </a>
</div>
"""

COURS_HTML = """
<html>
<body>
  <div class="c-faceplate__price">
    <span class="c-instrument c-instrument--last">59,42</span>
  </div>
  <ul>
    <li class="c-list-info__item c-list-info__item--fixed-width">
      <p>Dernier détachement</p>
      <p>18.06.24</p>
    </li>
    <li class="c-list-info__item c-list-info__item--has-picto c-list-info__item--fixed-width">
      <p>Prochain détachement</p>
      <p>19.06.25</p>
    </li>
    <li class="c-list-info__item c-list-info__item--has-picto c-list-info__item--fixed-width">
      <p class="c-list-info__label">Dividende</p>
      <p class="c-list-info__value u-color-big-stone">3,02 EUR</p>
    </li>
  </ul>
</body>
</html>
"""

COURS_HTML_NO_DIVIDEND = """
<html>
<body>
  <div class="c-faceplate__price">
    <span class="c-instrument c-instrument--last">145,20</span>
  </div>
  <ul>
    <li class="c-list-info__item"><p>PE</p><p>25,4</p></li>
  </ul>
</body>
</html>
"""

COURS_HTML_PRICE_ONLY = """
<html>
<body>
  <span class="c-instrument c-instrument--last">100,00</span>
</body>
</html>
"""

CONSENSUS_HTML = """
<html>
<body>
  <div class="c-median-gauge__tooltip">Achat Fort</div>
</body>
</html>
"""

CONSENSUS_HTML_EMPTY = "<html><body><p>Pas de consensus disponible</p></body></html>"

FINANCIALS_HTML = """
<html>
<body>
  <div class="c-company-statements">
    <table>
      <thead>
        <tr><th>Indicateur</th><th>2022</th><th>2023</th><th>2024</th></tr>
      </thead>
      <tbody>
        <tr><td>CA (M€)</td><td>263 310</td><td>218 945</td><td>195 520</td></tr>
        <tr><td>EBIT</td><td>40 200</td><td>32 100</td><td>28 400</td></tr>
        <tr><td>Résultat net</td><td>20 526</td><td>18 020</td><td>15 800</td></tr>
      </tbody>
    </table>
  </div>
</body>
</html>
"""

FINANCIALS_HTML_NO_TABLE = "<html><body><p>Données non disponibles</p></body></html>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(text: str, status_code: int = 200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        r.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return r


def _make_provider(tmp_path, exchange_filter="Euronext", ttl=86400):
    return BoursoramaProvider(
        cache_dir=str(tmp_path),
        cache_ttl_seconds=ttl,
        exchange_filter=exchange_filter,
    )


# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------

class TestParseFloat:
    def test_plain_number(self):
        assert _parse_float("59.42") == pytest.approx(59.42)

    def test_french_comma_decimal(self):
        assert _parse_float("59,42") == pytest.approx(59.42)

    def test_with_spaces(self):
        assert _parse_float("1 234,56") == pytest.approx(1234.56)

    def test_none_input(self):
        assert _parse_float(None) is None

    def test_non_numeric(self):
        assert _parse_float("N/A") is None

    def test_negative(self):
        assert _parse_float("-3,02") == pytest.approx(-3.02)


class TestParseFloatWithCurrency:
    def test_eur_value(self):
        assert _parse_float_with_currency("3,02 EUR", "EUR") == pytest.approx(3.02)

    def test_no_currency_match(self):
        assert _parse_float_with_currency("3,02 USD", "EUR") is None

    def test_embedded_in_text(self):
        assert _parse_float_with_currency("Dividende : 2,87 EUR versé", "EUR") == pytest.approx(2.87)


class TestParseDate:
    def test_dd_mm_yy(self):
        assert _parse_date("18.06.24") == "2024-06-18"

    def test_dd_slash_mm_slash_yyyy(self):
        assert _parse_date("18/06/2024") == "2024-06-18"

    def test_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_empty(self):
        assert _parse_date("") is None


# ---------------------------------------------------------------------------
# 2. _parse_search
# ---------------------------------------------------------------------------

class TestParseSearch:
    def test_finds_euronext_result(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_search(SEARCH_HTML_EURONEXT, "Euronext")
        assert result is not None
        code, nom, lien = result
        assert code == "1rTTE"
        assert "TotalEnergies" in nom
        assert lien == "https://www.boursorama.com/cours/1rTTE/"

    def test_exchange_filter_excludes_nyse(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_search(SEARCH_HTML_EURONEXT, "Euronext")
        code, _, _ = result
        assert code == "1rTTE"   # not "us-TTE"

    def test_no_filter_returns_first_result(self, tmp_path):
        p = _make_provider(tmp_path, exchange_filter=None)
        result = p._parse_search(SEARCH_HTML_EURONEXT, None)
        assert result is not None
        code, _, _ = result
        assert code == "1rTTE"   # first link in document order

    def test_returns_none_when_no_match(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_search(SEARCH_HTML_NO_RESULT, "Euronext")
        assert result is None

    def test_custom_exchange_filter(self, tmp_path):
        p = _make_provider(tmp_path, exchange_filter="XETRA")
        result = p._parse_search(SEARCH_HTML_MULTI_EXCHANGE, "XETRA")
        assert result is not None
        code, _, _ = result
        assert code == "xetra-AIR"

    def test_malformed_html_returns_none(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_search("<<< invalid >>>", "Euronext")
        assert result is None

    def test_finds_result_with_legacy_h4_markup(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_search(SEARCH_HTML_LEGACY_H4, "Euronext")
        assert result is not None
        code, nom, _ = result
        assert code == "1rOR"
        assert "Oréal" in nom


# ---------------------------------------------------------------------------
# 3. _parse_cours
# ---------------------------------------------------------------------------

class TestParseCours:
    def test_extracts_price(self, tmp_path):
        p = _make_provider(tmp_path)
        cours, _, _, _ = p._parse_cours(COURS_HTML, "FR0000")
        assert cours == pytest.approx(59.42)

    def test_extracts_dividend(self, tmp_path):
        p = _make_provider(tmp_path)
        _, dividende, _, _ = p._parse_cours(COURS_HTML, "FR0000")
        assert dividende == pytest.approx(3.02)

    def test_computes_rendement(self, tmp_path):
        p = _make_provider(tmp_path)
        cours, dividende, rendement, _ = p._parse_cours(COURS_HTML, "FR0000")
        expected = round(3.02 / 59.42 * 100, 4)
        assert rendement == pytest.approx(expected)

    def test_extracts_dividend_date(self, tmp_path):
        p = _make_provider(tmp_path)
        _, _, _, last_div_date = p._parse_cours(COURS_HTML, "FR0000")
        assert last_div_date == "2024-06-18"

    def test_no_dividend_returns_none(self, tmp_path):
        p = _make_provider(tmp_path)
        cours, dividende, rendement, date = p._parse_cours(COURS_HTML_NO_DIVIDEND, "FR0000")
        assert cours == pytest.approx(145.20)
        assert dividende is None
        assert rendement is None
        assert date is None

    def test_price_only_html(self, tmp_path):
        p = _make_provider(tmp_path)
        cours, _, _, _ = p._parse_cours(COURS_HTML_PRICE_ONLY, "FR0000")
        assert cours == pytest.approx(100.0)

    def test_empty_html_returns_all_none(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_cours("<html></html>", "FR0000")
        assert result == (None, None, None, None)


# ---------------------------------------------------------------------------
# 4. _parse_consensus
# ---------------------------------------------------------------------------

class TestParseConsensus:
    def test_extracts_consensus(self, tmp_path):
        p = _make_provider(tmp_path)
        assert p._parse_consensus(CONSENSUS_HTML) == "Achat Fort"

    def test_returns_none_when_absent(self, tmp_path):
        p = _make_provider(tmp_path)
        assert p._parse_consensus(CONSENSUS_HTML_EMPTY) is None


# ---------------------------------------------------------------------------
# 5. _parse_financials
# ---------------------------------------------------------------------------

class TestParseFinancials:
    def test_extracts_ca_and_rn(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_financials(FINANCIALS_HTML, "FR0000")
        assert len(result) == 3
        assert result[0]["annee"] == "2022"
        assert result[0]["ca"] == pytest.approx(263310.0)
        assert result[0]["rn"] == pytest.approx(20526.0)

    def test_computes_marge(self, tmp_path):
        p = _make_provider(tmp_path)
        result = p._parse_financials(FINANCIALS_HTML, "FR0000")
        assert result[0]["marge"] == pytest.approx(20526 / 263310 * 100, rel=1e-3)

    def test_no_table_returns_empty_list(self, tmp_path):
        p = _make_provider(tmp_path)
        assert p._parse_financials(FINANCIALS_HTML_NO_TABLE, "FR0000") == []


# ---------------------------------------------------------------------------
# 6. Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:
    def _write_cache(self, tmp_path, isin, data, age_seconds=0):
        path = tmp_path / f"boursorama_{isin}.json"
        path.write_text(json.dumps({
            "timestamp": time.time() - age_seconds,
            "data": data,
        }))

    def _minimal_quote_dict(self, isin="FR0000131104"):
        return {
            "isin": isin, "code_bourso": "1rTTE", "nom": "TotalEnergies",
            "lien": "https://www.boursorama.com/cours/1rTTE/",
            "cours": 59.42, "dividende": 3.02, "rendement": 5.08,
            "last_dividend_date": "2024-06-18", "consensus": "Achat Fort",
            "performance": [], "cached_at": "2024-01-01T00:00:00",
        }

    async def test_fresh_cache_skips_http(self, tmp_path):
        self._write_cache(tmp_path, "FR0000131104", self._minimal_quote_dict())
        p = _make_provider(tmp_path, ttl=86400)
        with patch.object(p, "_http_get") as mock_get:
            result = await p.get_quote("FR0000131104")
        mock_get.assert_not_called()
        assert result.cours == pytest.approx(59.42)

    async def test_expired_cache_triggers_fetch(self, tmp_path):
        self._write_cache(tmp_path, "FR0000131104", self._minimal_quote_dict(), age_seconds=90000)
        p = _make_provider(tmp_path, ttl=86400)

        responses = [
            SEARCH_HTML_EURONEXT,
            COURS_HTML,
            CONSENSUS_HTML,
            FINANCIALS_HTML,
        ]
        call_iter = iter(responses)
        with patch.object(p, "_http_get", side_effect=lambda url: next(call_iter)):
            result = await p.get_quote("FR0000131104")

        assert result.nom == "TotalEnergies SE"

    async def test_cache_miss_triggers_fetch(self, tmp_path):
        p = _make_provider(tmp_path)
        responses = [SEARCH_HTML_EURONEXT, COURS_HTML, CONSENSUS_HTML, FINANCIALS_HTML]
        call_iter = iter(responses)
        with patch.object(p, "_http_get", side_effect=lambda url: next(call_iter)):
            result = await p.get_quote("FR0000131104")

        assert result.code_bourso == "1rTTE"
        cache_file = tmp_path / "boursorama_FR0000131104.json"
        assert cache_file.exists()

    async def test_http_error_falls_back_to_stale_cache(self, tmp_path):
        self._write_cache(tmp_path, "FR0000131104", self._minimal_quote_dict(), age_seconds=90000)
        p = _make_provider(tmp_path, ttl=86400)
        with patch.object(p, "_http_get", side_effect=APIError("timeout")):
            result = await p.get_quote("FR0000131104")
        assert result.cours == pytest.approx(59.42)

    async def test_http_error_no_cache_raises(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", side_effect=APIError("timeout")):
            with pytest.raises(APIError):
                await p.get_quote("FR0000131104")

    def test_invalidate_removes_cache_file(self, tmp_path):
        self._write_cache(tmp_path, "FR0000131104", self._minimal_quote_dict())
        p = _make_provider(tmp_path)
        p.invalidate_cache("FR0000131104")
        assert not (tmp_path / "boursorama_FR0000131104.json").exists()

    def test_invalidate_nonexistent_is_silent(self, tmp_path):
        p = _make_provider(tmp_path)
        p.invalidate_cache("FR9999999999")   # should not raise


# ---------------------------------------------------------------------------
# 7. get_quote integration
# ---------------------------------------------------------------------------

class TestGetQuoteIntegration:
    async def test_full_quote_populated(self, tmp_path):
        p = _make_provider(tmp_path)
        responses = [SEARCH_HTML_EURONEXT, COURS_HTML, CONSENSUS_HTML, FINANCIALS_HTML]
        call_iter = iter(responses)
        with patch.object(p, "_http_get", side_effect=lambda url: next(call_iter)):
            q = await p.get_quote("FR0000131104")

        assert isinstance(q, BoursoramaQuote)
        assert q.isin == "FR0000131104"
        assert q.code_bourso == "1rTTE"
        assert q.cours == pytest.approx(59.42)
        assert q.dividende == pytest.approx(3.02)
        assert q.rendement is not None
        assert q.consensus == "Achat Fort"
        assert len(q.performance) == 3
        assert q.cached_at is not None

    async def test_isin_not_found_raises(self, tmp_path):
        p = _make_provider(tmp_path)
        with patch.object(p, "_http_get", return_value=SEARCH_HTML_NO_RESULT):
            with pytest.raises(APIError, match="Not found"):
                await p.get_quote("XX0000000000")

    async def test_http_calls_use_executor(self, tmp_path):
        """_http_get is called inside run_in_executor (non-blocking)."""
        p = _make_provider(tmp_path)
        calls = []

        def recording_get(url):
            calls.append(url)
            if "recherche" in url:
                return SEARCH_HTML_EURONEXT
            if "consensus" in url:
                return CONSENSUS_HTML
            if "chiffres" in url:
                return FINANCIALS_HTML
            return COURS_HTML

        with patch.object(p, "_http_get", side_effect=recording_get):
            await p.get_quote("FR0000131104")

        assert len(calls) == 4
        assert any("recherche" in c for c in calls)
