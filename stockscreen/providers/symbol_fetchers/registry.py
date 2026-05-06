"""Registry mapping category names to fetcher classes."""

from stockscreen.providers.symbol_fetchers.base import BaseSymbolFetcher
from stockscreen.providers.symbol_fetchers.wikipedia import (
    AEXFetcher,
    CAC40Fetcher,
    DAXFetcher,
    FTSE100Fetcher,
    Nasdaq100Fetcher,
    SBF120Fetcher,
    SP500Fetcher,
)

_REGISTRY: dict[str, type[BaseSymbolFetcher]] = {
    "sp500": SP500Fetcher,
    "nasdaq100": Nasdaq100Fetcher,
    "cac40": CAC40Fetcher,
    "sbf120": SBF120Fetcher,
    "dax": DAXFetcher,
    "ftse100": FTSE100Fetcher,
    "aex": AEXFetcher,
}


def build_fetchers(sources: list[str]) -> list[BaseSymbolFetcher]:
    """Instantiate fetchers for the given source names.

    Unknown names are logged and skipped.

    Args:
        sources: Ordered list of category names (e.g. ["sp500", "cac40"]).

    Returns:
        List of instantiated fetchers in the same order as *sources*.
    """
    import logging
    logger = logging.getLogger("stockscreen-server-v1")

    fetchers: list[BaseSymbolFetcher] = []
    for name in sources:
        cls = _REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Unknown symbol source '{name}' — skipping.")
        else:
            fetchers.append(cls())
    return fetchers


def available_sources() -> list[str]:
    """Return all registered source names."""
    return list(_REGISTRY)
