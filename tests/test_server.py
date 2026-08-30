import pandas as pd

from stockscreen import server
from stockscreen.persistence import Store


class T:
    pass


def test_legacy_false_positive_is_rejected_and_save_is_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "store", Store(tmp_path))
    monkeypatch.setattr(server.provider, "ticker", lambda symbol: T())
    monkeypatch.setattr(
        server.provider,
        "info",
        lambda *args: {
            "currentPrice": 10,
            "trailingPE": 500,
            "stamp": pd.Timestamp("2026-01-01"),
            "bad": float("nan"),
        },
    )
    result = server.run_stock_screen(
        "fundamental", {"symbols": ["JUNK"], "max_pe": 20}, save_result="saved"
    )
    assert not result["data"]["matches"] and result["data"]["rejected"][0]["symbol"] == "JUNK"
    assert server.store.load("result", "saved")


def test_unknown_criteria_fails():
    import pytest
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        server.run_stock_screen("technical", {"symbols": ["X"], "above_sma_20": True})
