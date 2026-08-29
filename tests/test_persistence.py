import pytest

from stockscreen.persistence import Store


def test_atomic_roundtrip_and_traversal(tmp_path):
    store = Store(tmp_path)
    store.save("watchlist", "safe", ["AAPL"])
    assert store.load("watchlist", "safe") == ["AAPL"]
    with pytest.raises(ValueError):
        store.save("watchlist", "../escape", [])
    assert store.delete("watchlist", "safe") and not store.delete("watchlist", "safe")
