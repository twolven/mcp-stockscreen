"""BoursoramaProvider — async scraper for Boursorama financial data.

Uses ISIN as the primary identifier. All HTTP calls run inside a thread-pool
executor so the event loop is never blocked.

Parsing strategy:
  - lxml.html  : CSS-selector queries on non-table HTML (search, cours, consensus)
  - pd.read_html: table extraction for chiffres-clés (pandas already a dep)

Cache: one JSON file per ISIN under ``cache_dir``, TTL configurable.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Optional

import lxml.html
import pandas as pd
import requests

from stockscreen.exceptions import APIError

logger = logging.getLogger("stockscreen-server-v1")

# Browser-like headers to avoid bot-detection redirects
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
    "Referer": "https://www.boursorama.com/",
}


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class BoursoramaQuote:
    """Financial data for one security, as scraped from Boursorama."""

    isin: str
    code_bourso: str
    nom: str
    lien: str
    cours: Optional[float] = None
    dividende: Optional[float] = None        # annual dividend amount (EUR)
    rendement: Optional[float] = None        # yield in %
    last_dividend_date: Optional[str] = None  # ISO YYYY-MM-DD
    consensus: Optional[str] = None          # analyst consensus label
    performance: list[dict] = field(default_factory=list)  # [{annee, ca, rn, marge}]
    cached_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class BoursoramaProvider:
    """Async Boursorama scraper with local JSON cache.

    Args:
        cache_dir: Directory where ``boursorama_{isin}.json`` files are stored.
        cache_ttl_seconds: Seconds before a cache entry is considered stale.
        exchange_filter: Keep only search results whose exchange text contains
            this string (case-insensitive). ``None`` accepts all exchanges.
            Examples: ``"Euronext"``, ``"XETRA"``, ``"NYSE"``, ``None``.
        timeout: Per-request HTTP timeout in seconds.
    """

    BASE_URL = "https://www.boursorama.com"
    _SEARCH_URL = f"{BASE_URL}/recherche/ajax?query={{isin}}&searchId="
    _CONSENSUS_URL = f"{BASE_URL}/cours/consensus/{{code}}/"
    _CHIFFRES_CLES_URL = f"{BASE_URL}/cours/societe/chiffres-cles/{{code}}/"

    def __init__(
        self,
        cache_dir: str,
        cache_ttl_seconds: float = 86400.0,
        exchange_filter: str | None = "Euronext",
        timeout: float = 10.0,
    ):
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl_seconds
        self._exchange_filter = exchange_filter
        self._timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        # Shared session: connection pooling + persistent cookies + headers
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_quote(self, query: str) -> BoursoramaQuote:
        """Fetch a fully populated quote.

        Args:
            query: ISIN (e.g. ``"FR0014000MR3"``), or short ticker without
                exchange suffix (e.g. ``"TTE"``, ``"AIR"``).
                Yahoo-format tickers with ``.PA`` / ``.DE`` suffixes are
                **not** supported — strip the suffix before calling.
                Short tickers can be ambiguous; use ``exchange_filter`` (set
                at construction time) to pick the right listing.

        Returns cached data when fresh. Falls back to stale cache on network
        error. Raises ``APIError`` if nothing is found and no cache exists.
        """
        cache = self._load_cache(query)
        if cache is not None and not self._is_expired(cache):
            return BoursoramaQuote(**cache["data"])

        try:
            quote = await self._fetch_all(query)
        except APIError:
            if cache is not None:
                logger.warning(f"[{query}] Fetch failed — serving stale cache.")
                return BoursoramaQuote(**cache["data"])
            raise

        self._save_cache(query, quote)
        return quote

    def invalidate_cache(self, query: str) -> None:
        """Remove the on-disk cache entry for *query* (ISIN or ticker)."""
        try:
            os.remove(self._cache_path(query))
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def _fetch_all(self, query: str) -> BoursoramaQuote:
        loop = asyncio.get_event_loop()

        # 1. Search → code_bourso, nom, lien
        search_html = await loop.run_in_executor(
            None, lambda: self._http_get(self._SEARCH_URL.format(isin=query))
        )
        found = self._parse_search(search_html, self._exchange_filter)
        if found is None:
            raise APIError(
                f"[{query}] Not found on Boursorama (exchange_filter={self._exchange_filter!r})"
            )
        code_bourso, nom, lien = found

        # 2. Cours page → cours, dividende, rendement, last_dividend_date
        cours_html = await loop.run_in_executor(
            None, lambda: self._http_get(lien)
        )
        cours, dividende, rendement, last_div_date = self._parse_cours(cours_html, query)

        # 3. Consensus
        consensus_html = await loop.run_in_executor(
            None, lambda: self._http_get(self._CONSENSUS_URL.format(code=code_bourso))
        )
        consensus = self._parse_consensus(consensus_html)

        # 4. Chiffres-clés (annual table → pandas)
        ck_html = await loop.run_in_executor(
            None, lambda: self._http_get(self._CHIFFRES_CLES_URL.format(code=code_bourso))
        )
        performance = self._parse_financials(ck_html, query)

        return BoursoramaQuote(
            isin=query,
            code_bourso=code_bourso,
            nom=nom,
            lien=lien,
            cours=cours,
            dividende=dividende,
            rendement=rendement,
            last_dividend_date=last_div_date,
            consensus=consensus,
            performance=performance,
            cached_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _http_get(self, url: str) -> str:
        """Synchronous GET — must be called inside run_in_executor."""
        try:
            resp = self._session.get(url, timeout=self._timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            raise APIError(f"HTTP error fetching {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_search(
        self, html: str, exchange_filter: str | None
    ) -> tuple[str, str, str] | None:
        """Return *(code_bourso, nom, lien)* or ``None`` if no match.

        Tries primary CSS selector first; falls back to any ``<a>`` whose
        ``href`` starts with ``/cours/`` if the primary selector yields nothing.
        """
        try:
            doc = lxml.html.fromstring(html)
        except Exception as exc:
            logger.warning(f"Search HTML parse error: {exc}")
            return None

        # Primary: structured search result links
        candidates = doc.cssselect("a.search__list-link")
        # Fallback: any /cours/ link (not products)
        if not candidates:
            candidates = [
                a for a in doc.cssselect("a[href]")
                if str(a.get("href", "")).startswith("/cours/")
                and "produits" not in str(a.get("href", ""))
            ]

        for link in candidates:
            href = str(link.get("href", "")).strip()
            # Skip structured products
            if not href or "produits" in href or "bourse/produits" in href:
                continue

            # Exchange filter — check search__item-content span (Boursorama's current markup)
            if exchange_filter:
                exchange_text = ""
                for sel in ("span.search__item-content", "p.search__item-content"):
                    els = link.cssselect(sel)
                    if els:
                        exchange_text = els[0].text_content()
                        break
                if exchange_filter.lower() not in exchange_text.lower():
                    continue

            # Company name — Boursorama changed <h4> to <span> in 2024; try both
            nom = ""
            for selector in (
                "span.search__item-title",
                "h4.search__item-title",
                "h4",
                "strong",
            ):
                els = link.cssselect(selector)
                if els:
                    nom = els[0].text_content().strip()
                    break
            if not nom:
                nom = link.text_content().strip().split("\n")[0].strip()

            code_bourso = href.rstrip("/").rsplit("/", 1)[-1]
            lien = f"{self.BASE_URL}{href}" if href.startswith("/") else href

            if code_bourso and nom:
                return code_bourso, nom, lien

        return None

    def _parse_cours(
        self, html: str, isin: str
    ) -> tuple[float | None, float | None, float | None, str | None]:
        """Return *(cours, dividende, rendement, last_dividend_date)*.

        Multiple fallback selectors are tried for each field so that minor
        HTML changes on Boursorama's side don't break everything at once.
        """
        try:
            doc = lxml.html.fromstring(html)
        except Exception as exc:
            logger.warning(f"[{isin}] Cours parse error: {exc}")
            return None, None, None, None

        # --- Price (multiple candidates) ---
        cours = None
        for selector in (
            "span.c-instrument--last",
            "[class*='instrument--last']",
            "span.last",
            "[data-ist-price]",
        ):
            els = doc.cssselect(selector)
            if els:
                cours = _parse_float(els[0].text_content())
                if cours is not None:
                    break

        # --- Dividend + date ---
        # Date and dividend amount are often in *different* <li> items, so we
        # collect them independently from all info-list items.
        dividende = None
        last_dividend_date = None

        # Pass 1: collect the last-detachment date from any <li> that mentions it
        for li in doc.cssselect(
            "li.c-list-info__item, li[class*='list-info__item']"
        ):
            text = li.text_content()
            if any(kw in text.lower() for kw in ("dernier", "détachement", "detachement")):
                for p in li.cssselect("p"):
                    d = _parse_date(p.text_content().strip())
                    if d:
                        last_dividend_date = d
                        break
            if last_dividend_date:
                break

        # Pass 2: find the dividend EUR amount
        # Strategy A: labelled value element
        for value_selector in (
            "p.c-list-info__value",
            "[class*='list-info__value']",
        ):
            for el in doc.cssselect(value_selector):
                text = el.text_content().strip()
                val = _parse_float_with_currency(text, "EUR")
                if val is not None and 0 < val < 100:
                    parent = el.getparent()
                    context = parent.text_content() if parent is not None else text
                    if any(kw in context.lower() for kw in ("dividende", "div")):
                        dividende = val
                        break
            if dividende is not None:
                break

        # Strategy B: scan <li> items for any EUR amount (fallback)
        if dividende is None:
            for li in doc.cssselect(
                "li.c-list-info__item--has-picto, li.c-list-info__item"
            ):
                val = _parse_float_with_currency(li.text_content(), "EUR")
                if val is not None and 0 < val < 100:
                    dividende = val
                    break

        # --- Rendement: always recompute from rate/price for consistency ---
        rendement = None
        if dividende and cours:
            rendement = round(dividende / cours * 100, 4)

        return cours, dividende, rendement, last_dividend_date

    def _parse_consensus(self, html: str) -> str | None:
        """Extract the analyst consensus label. Returns ``None`` if absent."""
        try:
            doc = lxml.html.fromstring(html)
        except Exception as exc:
            logger.warning(f"Consensus parse error: {exc}")
            return None

        for selector in (
            "div.c-median-gauge__tooltip",
            "[class*='gauge__tooltip']",
            "[class*='consensus']",
        ):
            els = doc.cssselect(selector)
            if els:
                text = els[0].text_content().strip()
                if text:
                    return text

        return None

    def _parse_financials(self, html: str, isin: str) -> list[dict]:
        """Parse the chiffres-clés HTML table with pandas.

        Returns ``[{annee, ca, rn, marge}, ...]`` or ``[]`` on failure.
        """
        try:
            tables = pd.read_html(StringIO(html), decimal=",", thousands="\xa0")
        except Exception:
            try:
                tables = pd.read_html(StringIO(html))
            except Exception:
                return []

        for df in tables:
            result = _extract_financials_from_df(df)
            if result:
                logger.debug(f"[{isin}] Parsed {len(result)} years of financials.")
                return result

        logger.debug(f"[{isin}] No recognisable financials table found.")
        return []

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, isin: str) -> str:
        return os.path.join(self._cache_dir, f"boursorama_{isin}.json")

    def _load_cache(self, isin: str) -> dict | None:
        try:
            with open(self._cache_path(isin)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"[{isin}] Cache read error: {exc}")
            return None

    def _save_cache(self, isin: str, quote: BoursoramaQuote) -> None:
        payload = {
            "timestamp": time.time(),
            "data": {
                "isin": quote.isin,
                "code_bourso": quote.code_bourso,
                "nom": quote.nom,
                "lien": quote.lien,
                "cours": quote.cours,
                "dividende": quote.dividende,
                "rendement": quote.rendement,
                "last_dividend_date": quote.last_dividend_date,
                "consensus": quote.consensus,
                "performance": quote.performance,
                "cached_at": quote.cached_at,
            },
        }
        try:
            with open(self._cache_path(isin), "w") as f:
                json.dump(payload, f)
        except Exception as exc:
            logger.warning(f"[{isin}] Cache write error: {exc}")

    def _is_expired(self, cache: dict) -> bool:
        return time.time() - cache.get("timestamp", 0) > self._cache_ttl


# ---------------------------------------------------------------------------
# Module-level parsing helpers (also exported for testing)
# ---------------------------------------------------------------------------

def _parse_float(text: str | None) -> float | None:
    """Extract the first numeric value from *text*, handling French formatting."""
    if not text:
        return None
    # Normalise: non-breaking space + regular space as thousands sep, comma as decimal
    cleaned = (
        text.strip()
        .replace("\xa0", "")
        .replace("\u202f", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    m = re.search(r"[-+]?\d+\.?\d*", cleaned)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _parse_float_with_currency(text: str, currency: str) -> float | None:
    """Return the numeric value adjacent to *currency* in *text*, or ``None``."""
    pattern = rf"([\d\s\xa0,\.]+)\s*{re.escape(currency)}"
    m = re.search(pattern, text)
    if m:
        return _parse_float(m.group(1))
    return None


def _parse_date(text: str) -> str | None:
    """Parse common French date formats and return ISO ``YYYY-MM-DD``, or ``None``."""
    text = text.strip()
    for fmt in ("%d.%m.%y", "%d/%m/%Y", "%d/%m/%y", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_financials_from_df(df: pd.DataFrame) -> list[dict]:
    """Locate CA and RN rows in a pandas DataFrame and return annual dicts."""
    ca_row = rn_row = None

    for _, row in df.iterrows():
        label = str(row.iloc[0]).upper().strip()
        if ca_row is None and (
            "CHIFFRE" in label or re.match(r"^CA\b", label)
        ):
            ca_row = row
        if rn_row is None and (
            "RÉSULTAT NET" in label
            or "RESULTAT NET" in label
            or re.match(r"^RN\b", label)
        ):
            rn_row = row

    if ca_row is None or rn_row is None:
        return []

    results = []
    for i in range(1, len(df.columns)):
        try:
            year = str(df.columns[i])
            ca = _parse_float(str(ca_row.iloc[i]))
            rn = _parse_float(str(rn_row.iloc[i]))
            if ca is None or rn is None:
                continue
            marge = round(rn / ca * 100, 2) if ca else None
            results.append({"annee": year, "ca": ca, "rn": rn, "marge": marge})
        except (IndexError, ZeroDivisionError):
            continue

    return results
