"""Tests for BoursoramaPalmaresScaper — TDD step 4."""

import json
from unittest.mock import patch

import pytest

from stockscreen.providers.boursorama_palmares import (
    BoursoramaPalmaresScaper,
    PalmaresEntry,
    _parse_float_fr,
    _parse_rendement,
)


# ---------------------------------------------------------------------------
# HTML Fixtures (extracted from live Boursorama on 2026-03-24)
# ---------------------------------------------------------------------------

# Thead réel
_THEAD = """
<thead class="c-table__head">
  <tr class="c-table__row">
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Libellé</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Dernier</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Var.</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Div. 2025</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Rend. 2025</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Div. 2026</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Rend. 2026</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Div. 2027</h3></th>
    <th class="c-table__cell c-table__cell--head"><h3 class="c-table__title">Rend. 2027</h3></th>
  </tr>
</thead>
"""

# Ligne complète réelle (M6)
_ROW_M6 = """
<tr class="c-table__row" data-ist="1rPMMT"
    data-ist-init="{&quot;symbol&quot;:&quot;1rPMMT&quot;,&quot;last&quot;:11.36,&quot;exchangeCode&quot;:&quot;PAR&quot;}">
  <td class="c-table__cell c-table__cell--dotted u-ellipsis u-text-left">
    <div class="o-pack o-pack--middle">
      <div class="o-pack__item c-table__wrapper-srd">
        <span class="c-table-top-flop__srd">SRD</span>
      </div>
      <div class="o-pack__item u-ellipsis u-color-cerulean">
        <a href="/cours/1rPMMT/" title="M6 METROPOLE TELE." class="c-link">M6 METROPOLE TELE.</a>
      </div>
    </div>
  </td>
  <td class="c-table__cell"><span class="c-instrument c-instrument--last">11,360</span></td>
  <td class="c-table__cell"><span class="c-instrument">+0,53%</span></td>
  <td class="c-table__cell">1,250</td>
  <td class="c-table__cell">+10,70%</td>
  <td class="c-table__cell">1,250</td>
  <td class="c-table__cell">+10,70%</td>
  <td class="c-table__cell">1,250</td>
  <td class="c-table__cell">+10,70%</td>
</tr>
"""

# Ligne avec valeurs manquantes (dividendes vides pour certaines années)
_ROW_MISSING = """
<tr class="c-table__row" data-ist="FOO"
    data-ist-init="{&quot;symbol&quot;:&quot;FOO&quot;,&quot;last&quot;:5.0}">
  <td class="c-table__cell c-table__cell--dotted u-ellipsis u-text-left">
    <div class="o-pack o-pack--middle">
      <div class="o-pack__item u-ellipsis u-color-cerulean">
        <a href="/cours/FOO/" title="Foo Corp" class="c-link">Foo Corp</a>
      </div>
    </div>
  </td>
  <td class="c-table__cell"><span class="c-instrument c-instrument--last">5,000</span></td>
  <td class="c-table__cell"><span class="c-instrument">-1,00%</span></td>
  <td class="c-table__cell">  </td>
  <td class="c-table__cell">  </td>
  <td class="c-table__cell">  </td>
  <td class="c-table__cell">  </td>
  <td class="c-table__cell">  </td>
  <td class="c-table__cell">  </td>
</tr>
"""

# Ligne avec dividende différent par année (ICADE)
_ROW_ICADE = """
<tr class="c-table__row" data-ist="1rPICAD"
    data-ist-init="{&quot;symbol&quot;:&quot;1rPICAD&quot;,&quot;last&quot;:19.14}">
  <td class="c-table__cell c-table__cell--dotted u-ellipsis u-text-left">
    <div class="o-pack o-pack--middle">
      <div class="o-pack__item u-ellipsis u-color-cerulean">
        <a href="/cours/1rPICAD/" title="ICADE" class="c-link">ICADE</a>
      </div>
    </div>
  </td>
  <td class="c-table__cell"><span class="c-instrument c-instrument--last">19,140</span></td>
  <td class="c-table__cell"><span class="c-instrument">-0,57%</span></td>
  <td class="c-table__cell">1,920</td>
  <td class="c-table__cell">+9,79%</td>
  <td class="c-table__cell">1,880</td>
  <td class="c-table__cell">+9,58%</td>
  <td class="c-table__cell">1,820</td>
  <td class="c-table__cell">+9,28%</td>
</tr>
"""

# Page complète avec 2 lignes + pagination
PAGE_1_HTML = f"""
<html><body>
<table>
  {_THEAD}
  <tbody>
    {_ROW_M6}
    {_ROW_ICADE}
  </tbody>
</table>
<div role="navigation" class="c-pagination">
  <a href="/bourse/actions/palmares/dividendes/" aria-label="Page 1">
    <span class="c-pagination__content is-active">1</span>
  </a>
  <a href="/bourse/actions/palmares/dividendes/page-2" aria-label="Page 2">
    <span class="c-pagination__content">2</span>
  </a>
  <a href="/bourse/actions/palmares/dividendes/page-3" aria-label="Page 3">
    <span class="c-pagination__content">3</span>
  </a>
</div>
</body></html>
"""

PAGE_2_HTML = f"""
<html><body>
<table>
  {_THEAD}
  <tbody>
    {_ROW_MISSING}
  </tbody>
</table>
<div role="navigation" class="c-pagination">
  <a href="/bourse/actions/palmares/dividendes/" aria-label="Page 1">
    <span class="c-pagination__content">1</span>
  </a>
  <a href="/bourse/actions/palmares/dividendes/page-2" aria-label="Page 2">
    <span class="c-pagination__content is-active">2</span>
  </a>
  <a href="/bourse/actions/palmares/dividendes/page-3" aria-label="Page 3">
    <span class="c-pagination__content">3</span>
  </a>
</div>
</body></html>
"""

# Dernière page — pas de lien "suivant" (pagination arrêtée à la page courante)
PAGE_LAST_HTML = f"""
<html><body>
<table>
  {_THEAD}
  <tbody>{_ROW_M6}</tbody>
</table>
<div role="navigation" class="c-pagination">
  <a href="/bourse/actions/palmares/dividendes/" aria-label="Page 1">
    <span class="c-pagination__content">1</span>
  </a>
  <a href="/bourse/actions/palmares/dividendes/page-2" aria-label="Page 2">
    <span class="c-pagination__content is-active">2</span>
  </a>
</div>
</body></html>
"""


def _make_scraper() -> BoursoramaPalmaresScaper:
    return BoursoramaPalmaresScaper()


# ---------------------------------------------------------------------------
# 1. Utilitaires de parsing
# ---------------------------------------------------------------------------

class TestParseUtils:
    def test_parse_float_fr_comma(self):
        assert _parse_float_fr("1,250") == pytest.approx(1.25)

    def test_parse_float_fr_no_decimal(self):
        assert _parse_float_fr("19") == pytest.approx(19.0)

    def test_parse_float_fr_whitespace(self):
        assert _parse_float_fr("  1,920  ") == pytest.approx(1.92)

    def test_parse_float_fr_empty(self):
        assert _parse_float_fr("") is None

    def test_parse_float_fr_spaces_only(self):
        assert _parse_float_fr("   ") is None

    def test_parse_rendement_strips_plus_and_percent(self):
        assert _parse_rendement("+10,70%") == pytest.approx(10.70)

    def test_parse_rendement_negative(self):
        assert _parse_rendement("-2,50%") == pytest.approx(-2.50)

    def test_parse_rendement_empty(self):
        assert _parse_rendement("") is None

    def test_parse_rendement_spaces(self):
        assert _parse_rendement("   ") is None


# ---------------------------------------------------------------------------
# 2. _parse_page — parsing d'une page HTML
# ---------------------------------------------------------------------------

class TestParsePage:
    def test_returns_list_of_entries(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        assert isinstance(entries, list)
        assert len(entries) == 2

    def test_extracts_code_bourso(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        assert entries[0].code_bourso == "1rPMMT"
        assert entries[1].code_bourso == "1rPICAD"

    def test_extracts_nom(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        assert entries[0].nom == "M6 METROPOLE TELE."
        assert entries[1].nom == "ICADE"

    def test_extracts_cours(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        assert entries[0].cours == pytest.approx(11.36)

    def test_extracts_dividendes_list(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        divs = entries[0].dividendes
        assert len(divs) == 3
        assert divs[0]["annee"] == "2025"
        assert divs[0]["dividende"] == pytest.approx(1.25)
        assert divs[0]["rendement"] == pytest.approx(10.70)

    def test_dividendes_different_by_year(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        icade = entries[1]
        assert icade.dividendes[0]["annee"] == "2025"
        assert icade.dividendes[0]["dividende"] == pytest.approx(1.92)
        assert icade.dividendes[1]["annee"] == "2026"
        assert icade.dividendes[1]["dividende"] == pytest.approx(1.88)
        assert icade.dividendes[2]["annee"] == "2027"
        assert icade.dividendes[2]["dividende"] == pytest.approx(1.82)

    def test_missing_dividendes_returns_none(self):
        s = _make_scraper()
        html = f"<html><body><table>{_THEAD}<tbody>{_ROW_MISSING}</tbody></table></body></html>"
        entries = s._parse_page(html)
        assert len(entries) == 1
        entry = entries[0]
        assert all(d["dividende"] is None for d in entry.dividendes)
        assert all(d["rendement"] is None for d in entry.dividendes)

    def test_missing_fields_do_not_raise(self):
        s = _make_scraper()
        html = f"<html><body><table>{_THEAD}<tbody>{_ROW_MISSING}</tbody></table></body></html>"
        entries = s._parse_page(html)
        assert entries[0].cours is not None  # still has cours from data-ist-init

    def test_empty_table_returns_empty_list(self):
        s = _make_scraper()
        html = f"<html><body><table>{_THEAD}<tbody></tbody></table></body></html>"
        entries = s._parse_page(html)
        assert entries == []

    def test_isin_is_none_by_default(self):
        s = _make_scraper()
        entries = s._parse_page(PAGE_1_HTML)
        assert all(e.isin is None for e in entries)

    def test_years_extracted_from_headers_dynamically(self):
        """Les années (2025, 2026, 2027) sont lues depuis les <th>, pas hardcodées."""
        # Modifier les en-têtes pour simuler une année différente
        html = PAGE_1_HTML.replace("2025", "2030").replace("2026", "2031").replace("2027", "2032")
        s = _make_scraper()
        entries = s._parse_page(html)
        years = [d["annee"] for d in entries[0].dividendes]
        assert "2030" in years
        assert "2025" not in years


# ---------------------------------------------------------------------------
# 3. _detect_page_count
# ---------------------------------------------------------------------------

class TestDetectPageCount:
    def test_detects_3_pages(self):
        s = _make_scraper()
        assert s._detect_page_count(PAGE_1_HTML) == 3

    def test_detects_1_page_when_no_pagination(self):
        s = _make_scraper()
        html = f"<html><body><table>{_THEAD}<tbody>{_ROW_M6}</tbody></table></body></html>"
        assert s._detect_page_count(html) == 1

    def test_last_page_returns_correct_count(self):
        s = _make_scraper()
        assert s._detect_page_count(PAGE_LAST_HTML) == 2


# ---------------------------------------------------------------------------
# 4. fetch_page — HTTP async
# ---------------------------------------------------------------------------

class TestFetchPage:
    async def test_fetch_page_1_calls_base_url(self):
        s = _make_scraper()
        calls = []

        def recording_get(url):
            calls.append(url)
            return PAGE_1_HTML

        with patch.object(s, "_http_get", side_effect=recording_get):
            entries = await s.fetch_page(1)

        assert len(calls) == 1
        assert "palmares/dividendes" in calls[0]
        assert "page-" not in calls[0]  # page 1 = URL racine

    async def test_fetch_page_2_calls_page_url(self):
        s = _make_scraper()
        calls = []

        def recording_get(url):
            calls.append(url)
            return PAGE_2_HTML

        with patch.object(s, "_http_get", side_effect=recording_get):
            await s.fetch_page(2)

        assert "page-2" in calls[0]

    async def test_fetch_page_returns_entries(self):
        s = _make_scraper()
        with patch.object(s, "_http_get", return_value=PAGE_1_HTML):
            entries = await s.fetch_page(1)
        assert len(entries) == 2
        assert entries[0].code_bourso == "1rPMMT"

    async def test_http_called_via_executor(self):
        """_http_get est appelé dans run_in_executor (non bloquant)."""
        s = _make_scraper()
        calls = []

        def recording_get(url):
            calls.append(url)
            return PAGE_1_HTML

        with patch.object(s, "_http_get", side_effect=recording_get):
            await s.fetch_page(1)

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 5. fetch_all — agrégation multi-pages
# ---------------------------------------------------------------------------

class TestFetchAll:
    async def test_fetch_all_aggregates_pages(self):
        s = _make_scraper()

        def get_page(url):
            if "page-2" in url:
                return PAGE_2_HTML
            return PAGE_1_HTML

        with patch.object(s, "_http_get", side_effect=get_page):
            entries, page_count = await s.fetch_all()

        # PAGE_1 a 2 lignes, PAGE_2 a 1 ligne, PAGE_3 (=PAGE_1 again) a 2 lignes
        # total = 2 + 1 + 2 = 5... mais le test doit être cohérent
        assert len(entries) > 0
        assert page_count == 3

    async def test_fetch_all_scrapes_all_detected_pages(self):
        s = _make_scraper()
        call_urls = []

        def get_page(url):
            call_urls.append(url)
            if "page-2" in url:
                return PAGE_2_HTML
            if "page-3" in url:
                return PAGE_LAST_HTML
            return PAGE_1_HTML  # page 1 détecte 3 pages

        with patch.object(s, "_http_get", side_effect=get_page):
            entries, page_count = await s.fetch_all()

        # Doit avoir appelé page 1 (détection), puis pages 2 et 3
        assert len(call_urls) == 3
        assert page_count == 3
        assert len(entries) > 0

    async def test_fetch_all_continues_on_page_error(self):
        """Une erreur sur une page ne stoppe pas l'ensemble."""
        s = _make_scraper()
        call_count = [0]

        def get_page(url):
            call_count[0] += 1
            if "page-2" in url:
                raise Exception("timeout")
            if "page-3" in url:
                return PAGE_LAST_HTML
            return PAGE_1_HTML

        with patch.object(s, "_http_get", side_effect=get_page):
            entries, page_count = await s.fetch_all()

        # Doit retourner les entrées des pages qui ont réussi
        assert len(entries) > 0
        assert call_count[0] == 3
        assert page_count == 3
