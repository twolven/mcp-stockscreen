"""Persistence for the Boursorama palmares dividendes snapshot."""

import json
import logging
import os
from dataclasses import asdict, dataclass

from stockscreen.providers.boursorama_palmares import PalmaresEntry

logger = logging.getLogger("stockscreen-server-v1")


@dataclass
class PalmaresSnapshot:
    """Full palmares snapshot with metadata.

    Fields:
        fetched_at:    ISO datetime of the last successful scrape.
        page_count:    Number of pages scraped.
        total_entries: Number of entries in this snapshot.
        entries:       List of :class:`PalmaresEntry`.
    """

    fetched_at: str
    page_count: int
    total_entries: int
    entries: list[PalmaresEntry]


class PalmaresStore:
    """Read/write a :class:`PalmaresSnapshot` to a single JSON file.

    Args:
        base_path: Root data directory (same as ``DEFAULT_DATA_PATH``).
                   The snapshot is stored at ``{base_path}/palmares/palmares_dividendes.json``.
    """

    def __init__(self, base_path: str):
        self._base_path = base_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, snapshot: PalmaresSnapshot) -> None:
        """Persist the snapshot, overwriting any previous file."""
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        data = {
            "fetched_at": snapshot.fetched_at,
            "page_count": snapshot.page_count,
            "total_entries": snapshot.total_entries,
            "entries": [asdict(e) for e in snapshot.entries],
        }
        try:
            with open(self._path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"PalmaresStore.save error: {exc}")

    def load(self) -> PalmaresSnapshot | None:
        """Load the snapshot from disk. Returns None if absent or corrupted."""
        try:
            with open(self._path(), encoding="utf-8") as f:
                data = json.load(f)
            entries = [PalmaresEntry(**e) for e in data["entries"]]
            return PalmaresSnapshot(
                fetched_at=data["fetched_at"],
                page_count=data["page_count"],
                total_entries=data["total_entries"],
                entries=entries,
            )
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"PalmaresStore.load error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self) -> str:
        return os.path.join(self._base_path, "palmares", "palmares_dividendes.json")
