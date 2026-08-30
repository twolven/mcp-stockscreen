import pytest

from stockscreen.persistence import Store


def test_atomic_roundtrip_and_traversal(tmp_path):
    store = Store(tmp_path)
    store.save("watchlist", "safe", ["AAPL"])
    assert store.load("watchlist", "safe") == ["AAPL"]
    with pytest.raises(ValueError):
        store.save("watchlist", "../escape", [])
    assert store.delete("watchlist", "safe") and not store.delete("watchlist", "safe")


def test_invalid_json_and_nonserializable_values(tmp_path):
    store = Store(tmp_path)
    path = store._path("result", "broken")
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load("result", "broken")
    with pytest.raises(TypeError):
        store.save("result", "bad", {"value": object()})
    assert not list(path.parent.glob(".tmp-*"))


def test_save_uses_atomic_replace(tmp_path, monkeypatch):
    import os

    calls = []
    real_replace = os.replace

    def spy(source, target):
        calls.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr("stockscreen.persistence.os.replace", spy)
    Store(tmp_path).save("result", "atomic", {"ok": True})
    assert len(calls) == 1
