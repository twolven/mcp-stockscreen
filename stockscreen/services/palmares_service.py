"""Palmares service — orchestrates scraping, caching and filtering."""

import datetime
import logging

from stockscreen.providers.boursorama_palmares import BoursoramaPalmaresScaper
from stockscreen.store.palmares_store import PalmaresSnapshot, PalmaresStore

logger = logging.getLogger("stockscreen-server-v1")


class PalmaresService:
    """Fetch, cache, and filter the Boursorama dividend palmares.

    Args:
        scraper:           :class:`BoursoramaPalmaresScaper` instance.
        store:             :class:`PalmaresStore` instance.
        cache_ttl_seconds: Seconds before the cached snapshot is considered stale.
    """

    def __init__(
        self,
        scraper: BoursoramaPalmaresScaper,
        store: PalmaresStore,
        cache_ttl_seconds: float = 86400.0,
    ):
        self._scraper = scraper
        self._store = store
        self._cache_ttl = cache_ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        min_rendement: float | None = None,
        max_rendement: float | None = None,
        nom_contains: str | None = None,
        limit: int = 100,
    ) -> PalmaresSnapshot:
        """Return a (possibly filtered) snapshot.

        The snapshot is fetched from the scraper only when the cache is
        absent or expired. Entries are sorted by best rendement (descending,
        None values last) before filtering.

        Args:
            min_rendement: Keep only entries whose best rendement ≥ this value.
            max_rendement: Keep only entries whose best rendement ≤ this value.
            nom_contains:  Case-insensitive substring match on ``nom``.
            limit:         Maximum number of entries to return (default 100).

        Returns:
            :class:`PalmaresSnapshot` with ``total_entries`` reflecting the
            *unfiltered* count and ``entries`` containing the filtered/limited
            results.
        """
        snapshot = await self._load_or_fetch()
        total = len(snapshot.entries)

        entries = self._sort(snapshot.entries)
        entries = self._filter(entries, min_rendement, max_rendement, nom_contains)
        entries = entries[:limit]

        return PalmaresSnapshot(
            fetched_at=snapshot.fetched_at,
            page_count=snapshot.page_count,
            total_entries=total,
            entries=entries,
        )

    async def refresh(self) -> PalmaresSnapshot:
        """Force a new scrape regardless of cache freshness."""
        return await self._fetch_and_save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_or_fetch(self) -> PalmaresSnapshot:
        cached = self._store.load()
        if cached is not None and self._is_fresh(cached):
            return cached
        return await self._fetch_and_save()

    async def _fetch_and_save(self) -> PalmaresSnapshot:
        entries, page_count = await self._scraper.fetch_all()
        snapshot = PalmaresSnapshot(
            fetched_at=datetime.datetime.now().isoformat(),
            page_count=page_count,
            total_entries=len(entries),
            entries=entries,
        )
        self._store.save(snapshot)
        return snapshot

    def _is_fresh(self, snapshot: PalmaresSnapshot) -> bool:
        try:
            fetched = datetime.datetime.fromisoformat(snapshot.fetched_at)
            age = (datetime.datetime.now() - fetched).total_seconds()
            return age < self._cache_ttl
        except Exception:
            return False

    def _sort(self, entries):
        def _key(e):
            rend = self._best_rendement(e)
            return (rend is None, -(rend or 0))

        return sorted(entries, key=_key)

    def _filter(self, entries, min_rendement, max_rendement, nom_contains):
        result = []
        nom_lower = nom_contains.lower() if nom_contains else None
        for e in entries:
            rend = self._best_rendement(e)
            if min_rendement is not None and (rend is None or rend < min_rendement):
                continue
            if max_rendement is not None and (rend is None or rend > max_rendement):
                continue
            if nom_lower and nom_lower not in e.nom.lower():
                continue
            result.append(e)
        return result

    @staticmethod
    def _best_rendement(entry) -> float | None:
        """Return the highest rendement across all dividend years, or None."""
        values = [
            d["rendement"]
            for d in entry.dividendes
            if d.get("rendement") is not None
        ]
        return max(values) if values else None
