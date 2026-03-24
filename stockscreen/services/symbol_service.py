"""SymbolService — cache, refresh, and background scheduling for symbol lists."""

import asyncio
import json
import logging
import os
import time

from stockscreen.exceptions import APIError, ValidationError
from stockscreen.providers.symbol_fetchers.base import BaseSymbolFetcher

logger = logging.getLogger("stockscreen-server-v1")


class SymbolService:
    """Manages symbol lists fetched from external index sources.

    Responsibilities:
      - Resolve a category name (e.g. "cac40") to a list of ticker strings.
      - Cache results on disk; refresh when stale.
      - Support forced refresh (on-demand or at startup).
      - Run a background tick that refreshes expired categories periodically.

    Args:
        fetchers: One fetcher per category. Duplicate names are not allowed.
        cache_dir: Directory where per-category JSON cache files are stored.
        refresh_interval_hours: Age threshold after which a cache is stale.
    """

    def __init__(
        self,
        fetchers: list[BaseSymbolFetcher],
        cache_dir: str,
        refresh_interval_hours: float = 24.0,
    ):
        self._fetchers: dict[str, BaseSymbolFetcher] = {f.name: f for f in fetchers}
        self._cache_dir = cache_dir
        self._refresh_interval = refresh_interval_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_categories(self) -> list[str]:
        """Return all registered category names."""
        return list(self._fetchers)

    async def get(self, category: str) -> list[str]:
        """Return the symbol list for *category*.

        Hits the on-disk cache when fresh; fetches otherwise.
        Falls back to a stale cache if the fetch fails.

        Raises:
            ValidationError: Unknown category.
            APIError: Fetch failed and no cache is available.
        """
        self._require_known(category)

        cache = self._load_cache(category)

        if cache is None or self._is_expired(cache):
            try:
                await self._fetch_and_cache(category)
                cache = self._load_cache(category)
            except Exception as exc:
                if cache is not None:
                    logger.warning(
                        f"[{category}] Fetch failed ({exc}), serving stale cache."
                    )
                else:
                    raise

        return [r["symbol"] for r in (cache or {}).get("symbols", [])]

    async def refresh(self, category: str | None = None) -> dict:
        """Force-refresh one or all categories.

        Args:
            category: Specific category to refresh, or *None* for all.

        Returns:
            Dict mapping category name → symbol count fetched,
            or → {"error": message} on fetch failure.

        Raises:
            ValidationError: *category* is not registered.
        """
        if category is not None:
            self._require_known(category)
            categories = [category]
        else:
            categories = list(self._fetchers)

        results: dict = {}
        for cat in categories:
            try:
                count = await self._fetch_and_cache(cat)
                results[cat] = count
            except Exception as exc:
                logger.error(f"[{cat}] Refresh failed: {exc}")
                results[cat] = {"error": str(exc)}

        return results

    async def start_background_refresh(self, poll_interval_seconds: float = 3600.0) -> None:
        """Run forever, checking for stale caches every *poll_interval_seconds*.

        Intended to be launched as an asyncio background task.
        """
        while True:
            await asyncio.sleep(poll_interval_seconds)
            await self._run_background_tick()

    async def _run_background_tick(self) -> None:
        """Refresh every category whose cache has expired. Errors are logged, not raised."""
        for cat in self._fetchers:
            cache = self._load_cache(cat)
            if cache is None or self._is_expired(cache):
                try:
                    await self._fetch_and_cache(cat)
                    logger.info(f"[{cat}] Background refresh complete.")
                except Exception as exc:
                    logger.error(f"[{cat}] Background refresh failed: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_known(self, category: str) -> None:
        if category not in self._fetchers:
            known = ", ".join(sorted(self._fetchers)) or "(none)"
            raise ValidationError(
                f"Unknown symbol category '{category}'. Known: {known}"
            )

    def _cache_path(self, category: str) -> str:
        return os.path.join(self._cache_dir, f"{category}.json")

    def _load_cache(self, category: str) -> dict | None:
        try:
            with open(self._cache_path(category)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"[{category}] Cache read error: {exc}")
            return None

    def _is_expired(self, cache: dict) -> bool:
        return time.time() - cache.get("timestamp", 0) > self._refresh_interval

    async def _fetch_and_cache(self, category: str) -> int:
        """Fetch symbols and write them to the cache file. Returns symbol count."""
        fetcher = self._fetchers[category]
        records = await fetcher.fetch()
        data = {
            "timestamp": time.time(),
            "symbols": [r.to_dict() for r in records],
        }
        with open(self._cache_path(category), "w") as f:
            json.dump(data, f)
        logger.info(f"[{category}] Cached {len(records)} symbols.")
        return len(records)
