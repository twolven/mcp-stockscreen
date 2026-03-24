"""Configuration, paths, and logging setup for stockscreen."""

import logging
import os
import shutil

# Paths relative to this file (stockscreen/ package directory)
_package_dir = os.path.dirname(__file__)

DEFAULT_DATA_PATH = os.environ.get(
    "STOCKSCREEN_DATA_PATH",
    os.path.join(_package_dir, "data"),
)
DEFAULT_LOG_PATH = os.path.join(_package_dir, "stockscreen_v1.log")

# ---------------------------------------------------------------------------
# Symbol source configuration
# ---------------------------------------------------------------------------

# Comma-separated list of enabled index fetchers (order preserved).
# Override with STOCKSCREEN_SYMBOL_SOURCES env var.
SYMBOL_SOURCES: list[str] = [
    s.strip()
    for s in os.environ.get(
        "STOCKSCREEN_SYMBOL_SOURCES",
        "sp500,nasdaq100,cac40,sbf120,dax,ftse100,aex",
    ).split(",")
    if s.strip()
]

# Refresh symbol cache at server startup when cache is missing or expired.
# Set STOCKSCREEN_REFRESH_ON_STARTUP=false to disable.
REFRESH_ON_STARTUP: bool = (
    os.environ.get("STOCKSCREEN_REFRESH_ON_STARTUP", "true").lower() != "false"
)

# How many hours before a symbol cache entry is considered stale.
# Override with STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS env var.
SYMBOL_REFRESH_INTERVAL_HOURS: float = float(
    os.environ.get("STOCKSCREEN_SYMBOL_REFRESH_INTERVAL_HOURS", "24")
)

# ---------------------------------------------------------------------------
# Provider cache configuration
# ---------------------------------------------------------------------------

# Euronext ISIN↔ticker mapping cache TTL in seconds (default 7 days).
# Override with STOCKSCREEN_EURONEXT_CACHE_TTL env var.
EURONEXT_CACHE_TTL_SECONDS: float = float(
    os.environ.get("STOCKSCREEN_EURONEXT_CACHE_TTL", str(7 * 86400))
)

# Boursorama palmarès dividendes cache TTL in seconds (default 24h).
# Override with STOCKSCREEN_PALMARES_CACHE_TTL env var.
PALMARES_CACHE_TTL_SECONDS: float = float(
    os.environ.get("STOCKSCREEN_PALMARES_CACHE_TTL", "86400")
)


def migrate_legacy_data(target_path: str | None = None) -> None:
    """Migrate data from old CWD-based location to new package-relative location.

    Args:
        target_path: Override for the target directory. Defaults to DEFAULT_DATA_PATH.
    """
    if os.environ.get("STOCKSCREEN_DATA_PATH"):
        return  # User has explicitly set data path, skip migration

    old_data_path = os.path.join(os.getcwd(), "data")
    new_data_path = target_path or DEFAULT_DATA_PATH

    # Only migrate if old path exists, is different from new path, and new doesn't exist
    if (
        os.path.exists(old_data_path)
        and os.path.abspath(old_data_path) != os.path.abspath(new_data_path)
        and not os.path.exists(new_data_path)
    ):
        try:
            shutil.move(old_data_path, new_data_path)
        except Exception as e:
            print(f"[WARNING] Could not auto-migrate data: {e}")
            print(f"[WARNING] Please manually move data from {old_data_path} to {new_data_path}")


def get_logger() -> logging.Logger:
    """Return the application logger."""
    return logging.getLogger("stockscreen-server-v1")


def setup_logging() -> None:
    """Configure logging with file and stream handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(DEFAULT_LOG_PATH),
            logging.StreamHandler(),
        ],
    )
