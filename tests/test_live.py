import os

import pytest
import yfinance as yf

pytestmark = pytest.mark.skipif(os.getenv("YFINANCE_LIVE") != "1", reason="set YFINANCE_LIVE=1")


@pytest.mark.parametrize("symbol", ["AAPL", "SPY", "^GSPC", "BRK-B"])
def test_live_representative_quotes(symbol):
    info = yf.Ticker(symbol).get_info()
    assert isinstance(info, dict)
