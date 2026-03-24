"""Watchlist service — CRUD operations on watchlists."""

from stockscreen.models.schemas import StockSymbols, WatchlistName
from stockscreen.store.data_store import ScreenerDataStore


class WatchlistService:
    """Service for managing stock watchlists."""

    def __init__(self, store: ScreenerDataStore):
        self.store = store

    async def create(self, name: str, symbols: list[str]) -> dict:
        """Create a new watchlist."""
        validated_name = WatchlistName(name=name).name
        validated_symbols = StockSymbols(symbols=symbols).symbols
        self.store.save_watchlist(validated_name, validated_symbols)
        return {"message": f"Watchlist '{validated_name}' saved with {len(validated_symbols)} symbols"}

    async def get(self, name: str) -> dict:
        """Get a watchlist by name."""
        validated_name = WatchlistName(name=name).name
        symbols = self.store.load_watchlist(validated_name)
        if symbols is None:
            raise ValueError(f"Watchlist '{validated_name}' not found")
        return {"name": validated_name, "symbols": symbols}

    async def update(self, name: str, symbols: list[str]) -> dict:
        """Update an existing watchlist."""
        return await self.create(name, symbols)

    async def delete(self, name: str) -> dict:
        """Delete a watchlist."""
        validated_name = WatchlistName(name=name).name
        if not self.store.delete_watchlist(validated_name):
            raise ValueError(f"Watchlist '{validated_name}' not found")
        return {"message": f"Watchlist '{validated_name}' deleted"}

    async def dispatch(self, action: str, name: str, symbols: list[str] | None = None) -> dict:
        """Route an action to the appropriate method."""
        if action in ("create", "update"):
            if symbols is None:
                raise ValueError("symbols required for create/update")
            return await self.create(name, symbols)
        elif action == "get":
            return await self.get(name)
        elif action == "delete":
            return await self.delete(name)
        else:
            raise ValueError(f"Invalid action: {action}")
