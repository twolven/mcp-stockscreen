"""Boursorama palmarès dividendes scraper — async, multi-page."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import requests
from lxml import html

logger = logging.getLogger("stockscreen-server-v1")

_BASE_URL = "https://www.boursorama.com"
_PALMARES_URL = f"{_BASE_URL}/bourse/actions/palmares/dividendes/"
_PAGE_URL = f"{_PALMARES_URL}page-{{page}}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
    "Referer": "https://www.boursorama.com/",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float_fr(text: str) -> float | None:
    """Parse a French-formatted float string ('1,250' → 1.25). Returns None if empty."""
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned.replace(",", ".").replace("\xa0", "").replace(" ", ""))
    except ValueError:
        return None


def _parse_rendement(text: str) -> float | None:
    """Parse a French rendement string ('+10,70%' → 10.70). Returns None if empty."""
    cleaned = text.strip().replace("%", "").replace("+", "").replace("\xa0", "")
    if not cleaned:
        return None
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PalmaresEntry:
    """One row from the Boursorama dividend palmares table.

    Fields:
        code_bourso:  Boursorama internal code (e.g. ``"1rPMMT"``).
        nom:          Company name (e.g. ``"M6 METROPOLE TELE."``).
        cours:        Last price in EUR, or None.
        dividendes:   List of dicts ``{annee, dividende, rendement}`` per year,
                      ordered from oldest to most recent.
        isin:         ISIN if enriched externally, else None.
    """

    code_bourso: str
    nom: str
    cours: float | None
    dividendes: list[dict] = field(default_factory=list)
    isin: str | None = None


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class BoursoramaPalmaresScaper:
    """Async scraper for the Boursorama dividend palmares (multi-page).

    Args:
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_all(self) -> tuple[list[PalmaresEntry], int]:
        """Scrape all pages and return (entries, page_count)."""
        loop = asyncio.get_event_loop()

        # Page 1 — also used to detect total page count
        try:
            page1_html = await loop.run_in_executor(
                None, lambda: self._http_get(_PALMARES_URL)
            )
        except Exception as exc:
            logger.error(f"PalmaresScaper: failed to fetch page 1: {exc}")
            return [], 0

        total_pages = self._detect_page_count(page1_html)
        entries = self._parse_page(page1_html)

        # Remaining pages
        for page_num in range(2, total_pages + 1):
            try:
                page_html = await loop.run_in_executor(
                    None, lambda p=page_num: self._http_get(_PAGE_URL.format(page=p))
                )
                entries.extend(self._parse_page(page_html))
            except Exception as exc:
                logger.warning(f"PalmaresScaper: page {page_num} failed, skipping: {exc}")

        return entries, total_pages

    async def fetch_page(self, page: int) -> list[PalmaresEntry]:
        """Fetch and parse a single page (1-indexed)."""
        loop = asyncio.get_event_loop()
        url = _PALMARES_URL if page == 1 else _PAGE_URL.format(page=page)
        raw = await loop.run_in_executor(None, lambda: self._http_get(url))
        return self._parse_page(raw)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_page(self, page_html: str) -> list[PalmaresEntry]:
        """Parse one HTML page into a list of PalmaresEntry."""
        try:
            tree = html.fromstring(page_html)
        except Exception:
            return []

        # Extract year labels from <th> headers dynamically
        years = []
        for th in tree.cssselect("table thead th h3.c-table__title"):
            text = th.text_content().strip()
            m = re.search(r"Div\.\s*(\d{4})", text)
            if m:
                years.append(m.group(1))

        entries = []
        for row in tree.cssselect("table tbody tr"):
            entry = self._parse_row(row, years)
            if entry is not None:
                entries.append(entry)

        return entries

    def _parse_row(self, row, years: list[str]) -> PalmaresEntry | None:
        """Parse one <tr> into a PalmaresEntry. Returns None on unrecoverable error."""
        try:
            # code_bourso from data-ist attribute (most reliable)
            code_bourso = row.get("data-ist", "").strip()
            if not code_bourso:
                # fallback: extract from href
                a = row.cssselect("a[href*='/cours/']")
                if not a:
                    return None
                href = a[0].get("href", "")
                code_bourso = href.replace("/cours/", "").strip("/")

            # nom from <a title="..."> or link text
            a = row.cssselect("a[href*='/cours/']")
            if not a:
                return None
            nom = (a[0].get("title") or a[0].text_content()).strip()

            # cours from data-ist-init JSON (more reliable than scraped text)
            cours = None
            ist_init = row.get("data-ist-init", "")
            if ist_init:
                try:
                    ist_data = json.loads(ist_init)
                    cours = ist_data.get("last")
                except (json.JSONDecodeError, KeyError):
                    pass
            if cours is None:
                # fallback: scrape the last-price cell
                last_span = row.cssselect("span.c-instrument--last")
                if last_span:
                    cours = _parse_float_fr(last_span[0].text_content())

            # Dividend columns: pairs (Div.YYYY, Rend.YYYY) starting at td index 3
            tds = row.cssselect("td")
            dividendes = []
            for i, year in enumerate(years):
                div_idx = 3 + i * 2
                rend_idx = 4 + i * 2
                dividende = None
                rendement = None
                if div_idx < len(tds):
                    dividende = _parse_float_fr(tds[div_idx].text_content())
                if rend_idx < len(tds):
                    rendement = _parse_rendement(tds[rend_idx].text_content())
                dividendes.append({"annee": year, "dividende": dividende, "rendement": rendement})

            return PalmaresEntry(
                code_bourso=code_bourso,
                nom=nom,
                cours=cours,
                dividendes=dividendes,
            )

        except Exception as exc:
            logger.debug(f"PalmaresScaper: row parse error: {exc}")
            return None

    def _detect_page_count(self, page_html: str) -> int:
        """Return the total number of pages from the pagination element."""
        try:
            tree = html.fromstring(page_html)
            page_nums = []
            for a in tree.cssselect("div.c-pagination a[href*='palmares/dividendes']"):
                href = a.get("href", "")
                m = re.search(r"page-(\d+)", href)
                if m:
                    page_nums.append(int(m.group(1)))
            return max(page_nums) if page_nums else 1
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _http_get(self, url: str) -> str:
        """Synchronous HTTP GET — always called via run_in_executor."""
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.text
