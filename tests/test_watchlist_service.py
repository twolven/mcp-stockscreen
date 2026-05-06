"""Tests for stockscreen.services.watchlist module."""

import pytest

from stockscreen.services.watchlist import WatchlistService
from stockscreen.store.data_store import ScreenerDataStore


@pytest.fixture
def watchlist_service(tmp_path):
    store = ScreenerDataStore(base_path=str(tmp_path))
    return WatchlistService(store=store)


# ============================================================
# 1. Create
# ============================================================
class TestCreate:
    async def test_create_watchlist(self, watchlist_service):
        result = await watchlist_service.create("tech", ["AAPL", "MSFT"])
        assert result["message"] == "Watchlist 'tech' saved with 2 symbols"

    async def test_create_validates_name(self, watchlist_service):
        with pytest.raises(ValueError):
            await watchlist_service.create("-bad", ["AAPL"])

    async def test_create_validates_symbols(self, watchlist_service):
        with pytest.raises(ValueError):
            await watchlist_service.create("test", ["AA$L"])

    async def test_create_uppercases_symbols(self, watchlist_service):
        await watchlist_service.create("test", ["aapl", "msft"])
        result = await watchlist_service.get("test")
        assert result["symbols"] == ["AAPL", "MSFT"]


# ============================================================
# 2. Get
# ============================================================
class TestGet:
    async def test_get_existing(self, watchlist_service):
        await watchlist_service.create("test", ["AAPL"])
        result = await watchlist_service.get("test")
        assert result["name"] == "test"
        assert result["symbols"] == ["AAPL"]

    async def test_get_nonexistent(self, watchlist_service):
        with pytest.raises(ValueError, match="not found"):
            await watchlist_service.get("nonexistent")


# ============================================================
# 3. Update
# ============================================================
class TestUpdate:
    async def test_update_watchlist(self, watchlist_service):
        await watchlist_service.create("test", ["AAPL"])
        result = await watchlist_service.update("test", ["MSFT", "GOOG"])
        assert "2 symbols" in result["message"]

        got = await watchlist_service.get("test")
        assert got["symbols"] == ["MSFT", "GOOG"]


# ============================================================
# 4. Delete
# ============================================================
class TestDelete:
    async def test_delete_existing(self, watchlist_service):
        await watchlist_service.create("test", ["AAPL"])
        result = await watchlist_service.delete("test")
        assert "deleted" in result["message"]

    async def test_delete_nonexistent(self, watchlist_service):
        with pytest.raises(ValueError, match="not found"):
            await watchlist_service.delete("nonexistent")


# ============================================================
# 5. Dispatch
# ============================================================
class TestDispatch:
    async def test_dispatch_create(self, watchlist_service):
        result = await watchlist_service.dispatch(
            action="create", name="test", symbols=["AAPL"]
        )
        assert "saved" in result["message"]

    async def test_dispatch_get(self, watchlist_service):
        await watchlist_service.dispatch(action="create", name="test", symbols=["AAPL"])
        result = await watchlist_service.dispatch(action="get", name="test")
        assert result["symbols"] == ["AAPL"]

    async def test_dispatch_delete(self, watchlist_service):
        await watchlist_service.dispatch(action="create", name="test", symbols=["AAPL"])
        result = await watchlist_service.dispatch(action="delete", name="test")
        assert "deleted" in result["message"]

    async def test_dispatch_invalid_action(self, watchlist_service):
        with pytest.raises(ValueError, match="Invalid action"):
            await watchlist_service.dispatch(action="invalid", name="test")
