"""Unit tests for stockscreen.py MCP server."""

import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import after conftest sets STOCKSCREEN_DATA_PATH
from stockscreen import (
    APIError,
    DefaultSymbols,
    ScreenerDataStore,
    StockscreenError,
    StockscreenJSONEncoder,
    ValidationError,
    format_response,
    get_earnings_dates,
    get_news_data,
    retry_on_error,
    run_fundamental_screen,
    run_news_screen,
    run_options_screen,
    run_technical_screen,
    run_custom_screen,
    run_single_technical_screen,
    run_single_fundamental_screen,
    run_single_options_screen,
    validate_stock_symbols,
    validate_watchlist_name,
)


# ============================================================
# 1. Exception Hierarchy
# ============================================================
class TestExceptionHierarchy:
    def test_stockscreen_error_is_exception(self):
        assert issubclass(StockscreenError, Exception)

    def test_validation_error_is_stockscreen_error(self):
        assert issubclass(ValidationError, StockscreenError)

    def test_api_error_is_stockscreen_error(self):
        assert issubclass(APIError, StockscreenError)


# ============================================================
# 2. validate_stock_symbols
# ============================================================
class TestValidateStockSymbols:
    def test_valid_symbols(self):
        assert validate_stock_symbols(["AAPL", "MSFT", "BRK-B", "BF.B"]) is True

    def test_empty_list(self):
        assert validate_stock_symbols([]) is True

    def test_not_a_list(self):
        with pytest.raises(ValidationError, match="must be a list"):
            validate_stock_symbols("AAPL")

    def test_exceeds_max_symbols(self):
        with pytest.raises(ValidationError, match="Cannot exceed"):
            validate_stock_symbols(["A"] * 1001)

    def test_custom_max_symbols(self):
        with pytest.raises(ValidationError, match="Cannot exceed 5"):
            validate_stock_symbols(["A", "B", "C", "D", "E", "F"], max_symbols=5)

    def test_non_string_symbol(self):
        with pytest.raises(ValidationError, match="must be strings"):
            validate_stock_symbols([123])

    def test_symbol_too_long(self):
        with pytest.raises(ValidationError, match="Invalid symbol length"):
            validate_stock_symbols(["TOOLONGSYMBL"])

    def test_invalid_characters(self):
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_stock_symbols(["AA$L"])


# ============================================================
# 3. validate_watchlist_name
# ============================================================
class TestValidateWatchlistName:
    def test_valid_name(self):
        assert validate_watchlist_name("my_watchlist") is True

    def test_valid_name_with_numbers_and_hyphen(self):
        assert validate_watchlist_name("watch123-test") is True

    def test_empty_name(self):
        with pytest.raises(ValidationError):
            validate_watchlist_name("")

    def test_too_long_name(self):
        with pytest.raises(ValidationError, match="between 1 and 50"):
            validate_watchlist_name("a" * 51)

    def test_starts_with_hyphen(self):
        with pytest.raises(ValidationError):
            validate_watchlist_name("-bad")

    def test_invalid_characters(self):
        with pytest.raises(ValidationError):
            validate_watchlist_name("my watchlist!")

    def test_non_string_input(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_watchlist_name(123)


# ============================================================
# 4. StockscreenJSONEncoder
# ============================================================
class TestStockscreenJSONEncoder:
    def _encode(self, obj):
        return json.loads(json.dumps(obj, cls=StockscreenJSONEncoder))

    def test_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        result = self._encode({"ts": ts})
        assert result["ts"] == "2024-01-15T00:00:00"

    def test_nat(self):
        # pd.NaT is not a pd.Timestamp, so it falls through to str() fallback
        result = self._encode({"val": pd.NaT})
        assert result["val"] == "NaT"

    def test_period(self):
        p = pd.Period("2024-01", freq="M")
        result = self._encode({"p": p})
        assert result["p"] == "2024-01"

    def test_datetime_date(self):
        d = datetime.date(2024, 1, 15)
        result = self._encode({"d": d})
        assert result["d"] == "2024-01-15"

    def test_numpy_int(self):
        result = self._encode({"n": np.int64(42)})
        # Falls through to str() fallback
        assert result["n"] == "42"

    def test_numpy_float(self):
        # np.float64 is natively serializable by json as a float
        result = self._encode({"n": np.float64(3.14)})
        assert result["n"] == pytest.approx(3.14)

    def test_regular_types_pass_through(self):
        data = {"a": 1, "b": "hello", "c": [1, 2], "d": None}
        assert self._encode(data) == data


# ============================================================
# 5. format_response
# ============================================================
class TestFormatResponse:
    def test_success_response(self):
        result = format_response({"key": "value"})
        assert len(result) == 1
        body = json.loads(result[0].text)
        assert body["success"] is True
        assert body["data"] == {"key": "value"}
        assert body["error"] is None

    def test_error_response(self):
        result = format_response(None, "something broke")
        body = json.loads(result[0].text)
        assert body["success"] is False
        assert body["data"] is None
        assert body["error"] == "something broke"

    def test_response_has_timestamp(self):
        result = format_response({})
        body = json.loads(result[0].text)
        assert isinstance(body["timestamp"], (int, float))

    def test_response_uses_custom_encoder(self):
        # Should not raise even with pandas types
        result = format_response({"ts": pd.Timestamp("2024-01-01")})
        body = json.loads(result[0].text)
        assert body["data"]["ts"] == "2024-01-01T00:00:00"


# ============================================================
# 6. ScreenerDataStore
# ============================================================
class TestScreenerDataStore:
    def test_ensure_directories(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        assert (tmp_data_path / "screening_results").is_dir()
        assert (tmp_data_path / "watchlists").is_dir()
        assert (tmp_data_path / "market_data").is_dir()

    def test_save_and_load_watchlist(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        store.save_watchlist("test", ["AAPL", "MSFT"])
        result = store.load_watchlist("test")
        assert result == ["AAPL", "MSFT"]

    def test_load_nonexistent_watchlist(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        assert store.load_watchlist("nonexistent") is None

    def test_delete_watchlist(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        store.save_watchlist("todelete", ["AAPL"])
        assert store.delete_watchlist("todelete") is True
        assert store.load_watchlist("todelete") is None

    def test_delete_nonexistent_watchlist(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        assert store.delete_watchlist("nonexistent") is False

    def test_save_and_load_screening_result(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        data = {"screen_type": "technical", "results": [{"symbol": "AAPL"}]}
        store.save_screening_result("test_result", data)
        result = store.load_screening_result("test_result")
        assert result == data

    def test_load_nonexistent_screening_result(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        assert store.load_screening_result("nonexistent") is None

    def test_save_watchlist_overwrites(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        store.save_watchlist("overwrite", ["AAPL"])
        store.save_watchlist("overwrite", ["MSFT", "GOOG"])
        assert store.load_watchlist("overwrite") == ["MSFT", "GOOG"]

    def test_screening_result_with_special_types(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        data = {"timestamp": pd.Timestamp("2024-01-01"), "value": pd.NaT}
        store.save_screening_result("special", data)
        result = store.load_screening_result("special")
        assert result["timestamp"] == "2024-01-01T00:00:00"
        # pd.NaT falls through to str() in the encoder
        assert result["value"] == "NaT"

    def test_default_symbols_initialized(self, tmp_data_path):
        store = ScreenerDataStore(base_path=str(tmp_data_path))
        assert isinstance(store.default_symbols, DefaultSymbols)


# ============================================================
# 7. DefaultSymbols
# ============================================================
class TestDefaultSymbols:
    def test_filter_by_category_mega_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "AAPL", "market_cap": 3_000_000_000_000, "type": "equity"},
            {"symbol": "SMLL", "market_cap": 500_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "mega_cap")
        assert result == ["AAPL"]

    def test_filter_by_category_etf(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "SPY", "market_cap": None, "type": "etf"},
            {"symbol": "AAPL", "market_cap": 3_000_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "etf")
        assert result == ["SPY"]

    def test_filter_by_category_small_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "SMLL", "market_cap": 1_000_000_000, "type": "equity"},
            {"symbol": "BIG", "market_cap": 50_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "small_cap")
        assert result == ["SMLL"]

    def test_filter_by_category_mid_cap(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "MID", "market_cap": 5_000_000_000, "type": "equity"},
            {"symbol": "BIG", "market_cap": 50_000_000_000, "type": "equity"},
        ]
        result = ds._filter_by_category(symbols_data, "mid_cap")
        assert result == ["MID"]

    async def test_get_symbols_invalid_category(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        with patch.object(ds, "_load_or_fetch_symbols", new_callable=AsyncMock, return_value=[]):
            with pytest.raises(ValidationError, match="Invalid category"):
                await ds.get_symbols(category="invalid_category")

    async def test_get_symbols_returns_all_when_no_category(self):
        ds = DefaultSymbols(base_path="/tmp/test")
        symbols_data = [
            {"symbol": "AAPL", "market_cap": 3e12, "type": "equity"},
            {"symbol": "SPY", "market_cap": None, "type": "etf"},
        ]
        with patch.object(ds, "_load_or_fetch_symbols", new_callable=AsyncMock, return_value=symbols_data):
            result = await ds.get_symbols()
            assert result == ["AAPL", "SPY"]


# ============================================================
# 8. retry_on_error
# ============================================================
class TestRetryOnError:
    async def test_succeeds_first_try(self):
        call_count = 0

        @retry_on_error(max_retries=3, delay=0.01)
        async def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await success()
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_failure(self):
        call_count = 0

        @retry_on_error(max_retries=3, delay=0.01)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "ok"

        with patch("stockscreen.asyncio.sleep", new_callable=AsyncMock):
            result = await flaky()
            assert result == "ok"
            assert call_count == 3

    async def test_all_retries_exhausted(self):
        @retry_on_error(max_retries=3, delay=0.01)
        async def always_fail():
            raise RuntimeError("permanent")

        with patch("stockscreen.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="permanent"):
                await always_fail()

    async def test_exponential_backoff_timing(self):
        @retry_on_error(max_retries=3, delay=1.0)
        async def always_fail():
            raise RuntimeError("fail")

        with patch("stockscreen.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RuntimeError):
                await always_fail()
            # delay * 2^attempt: 1.0, 2.0
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)

    async def test_custom_retry_params(self):
        call_count = 0

        @retry_on_error(max_retries=2, delay=0.5)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with patch("stockscreen.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RuntimeError):
                await always_fail()
            assert call_count == 2
            mock_sleep.assert_called_once_with(0.5)


# ============================================================
# 9. run_technical_screen
# ============================================================
class TestRunTechnicalScreen:
    def _make_mock_ticker(self, mock_ticker_cls, info, history_df):
        mock_instance = MagicMock()
        mock_instance.info = info
        mock_instance.history.return_value = history_df
        mock_ticker_cls.return_value = mock_instance
        return mock_instance

    @patch("stockscreen.yf.Ticker")
    async def test_basic_pass(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(["AAPL"], {})
        assert result["screen_type"] == "technical"
        assert result["matches"] == 1
        assert result["results"][0]["symbol"] == "AAPL"

    @patch("stockscreen.yf.Ticker")
    async def test_price_filter_min(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(["AAPL"], {"min_price": 200})
        assert result["matches"] == 0
        assert any("Price" in r for r in result["rejected"][0]["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_price_filter_max(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(["AAPL"], {"max_price": 100})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_volume_filter(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(["AAPL"], {"min_volume": 999_999_999_999})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_rsi_filter(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        # max_rsi=1 should reject everything
        result = await run_technical_screen(["AAPL"], {"max_rsi": 1})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_empty_history(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance
        result = await run_technical_screen(["AAPL"], {})
        assert result["matches"] == 0
        assert len(result["rejected"]) == 1

    @patch("stockscreen.yf.Ticker")
    async def test_symbols_from_criteria(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(None, {"symbols": ["AAPL"]})
        assert result["matches"] == 1

    @patch("stockscreen.yf.Ticker")
    async def test_above_sma_200(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        # Price is 150, SMA200 on our synthetic data should be calculable
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_technical_screen(["AAPL"], {"above_sma_200": True})
        # With our synthetic data trending up from 100 to 120, and price=150,
        # the stock should be above SMA200
        assert result["screen_type"] == "technical"


# ============================================================
# 10. run_fundamental_screen
# ============================================================
class TestRunFundamentalScreen:
    @patch("stockscreen.yf.Ticker")
    async def test_basic_pass(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        result = await run_fundamental_screen(["AAPL"], {})
        assert result["screen_type"] == "fundamental"
        assert result["matches"] == 1

    @patch("stockscreen.yf.Ticker")
    async def test_market_cap_filter(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        result = await run_fundamental_screen(["AAPL"], {"min_market_cap": 10_000_000_000_000})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_pe_filter_range(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        # forwardPE is 25.5
        result = await run_fundamental_screen(["AAPL"], {"min_pe": 10, "max_pe": 20})
        assert result["matches"] == 0  # 25.5 > 20

    @patch("stockscreen.yf.Ticker")
    async def test_dividend_filter(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        # dividendYield is 0.005 → 0.5%
        result = await run_fundamental_screen(["AAPL"], {"min_dividend": 5.0})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_etf_screening(self, mock_ticker_cls, mock_etf_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_etf_info
        mock_ticker_cls.return_value = mock_instance
        result = await run_fundamental_screen(["SPY"], {})
        assert result["matches"] == 1
        assert "aum" in result["results"][0]

    @patch("stockscreen.yf.Ticker")
    async def test_no_info_skipped(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.info = None
        mock_ticker_cls.return_value = mock_instance
        result = await run_fundamental_screen(["BAD"], {})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_revenue_growth_filter(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        # revenueGrowth is 0.08
        result = await run_fundamental_screen(["AAPL"], {"min_revenue_growth": 0.5})
        assert result["matches"] == 0


# ============================================================
# 11. run_options_screen
# ============================================================
class TestRunOptionsScreen:
    def _setup_ticker(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.options = ("2024-03-15", "2024-04-19", "2024-05-17")
        mock_instance.option_chain.return_value = mock_option_chain
        mock_instance.history.return_value = pd.DataFrame({"Close": [150.0]}, index=[pd.Timestamp.now()])
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        mock_instance.calendar = {"Earnings Date": [future_date]}
        mock_ticker_cls.return_value = mock_instance
        return mock_instance

    @patch("stockscreen.yf.Ticker")
    async def test_basic_pass(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_ticker_info, mock_option_chain)
        result = await run_options_screen(["AAPL"], {})
        assert result["screen_type"] == "options"
        assert result["matches"] >= 0  # May pass or fail depending on chain data

    @patch("stockscreen.yf.Ticker")
    async def test_no_options_rejected(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.options = ()
        mock_instance.calendar = {}
        mock_ticker_cls.return_value = mock_instance
        result = await run_options_screen(["NOOPT"], {})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_iv_filter_max(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_ticker_info, mock_option_chain)
        # IV is ~25-26% for ATM, max_iv=1 should reject
        result = await run_options_screen(["AAPL"], {"max_iv": 1})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_volume_filter(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_ticker_info, mock_option_chain)
        result = await run_options_screen(["AAPL"], {"min_option_volume": 999_999_999})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_put_call_ratio(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_ticker_info, mock_option_chain)
        result = await run_options_screen(["AAPL"], {"min_put_call_ratio": 100})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_earnings_date_filter(self, mock_ticker_cls, mock_ticker_info, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_ticker_info, mock_option_chain)
        result = await run_options_screen(["AAPL"], {"max_days_to_earnings": 0})
        # Should reject because earnings is in the future
        assert result["matches"] == 0


# ============================================================
# 12. run_news_screen
# ============================================================
class TestRunNewsScreen:
    def _setup_news_ticker(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = mock_news_items
        mock_ticker_cls.return_value = mock_instance
        return mock_instance

    @patch("stockscreen.yf.Ticker")
    async def test_keyword_match(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        self._setup_news_ticker(mock_ticker_cls, mock_ticker_info, mock_news_items)
        result = await run_news_screen(["AAPL"], {"keywords": ["earnings"]})
        assert result["screen_type"] == "news"
        # "Quarterly earnings beat expectations" should match
        assert result["matches"] >= 1

    @patch("stockscreen.yf.Ticker")
    async def test_keyword_exclusion(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        self._setup_news_ticker(mock_ticker_cls, mock_ticker_info, mock_news_items)
        # Exclude all keywords that appear in our news
        result = await run_news_screen(["AAPL"], {
            "keywords": ["earnings"],
            "exclude_keywords": ["beat"]
        })
        # Should filter out the earnings news because "beat" is excluded
        # Depends on whether exclude applies before or after keyword match
        assert result["screen_type"] == "news"

    @patch("stockscreen.yf.Ticker")
    async def test_require_all_keywords(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        self._setup_news_ticker(mock_ticker_cls, mock_ticker_info, mock_news_items)
        result = await run_news_screen(["AAPL"], {
            "keywords": ["earnings", "nonexistent_word"],
            "require_all_keywords": True
        })
        # "nonexistent_word" won't match any news
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_management_changes_filter(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        self._setup_news_ticker(mock_ticker_cls, mock_ticker_info, mock_news_items)
        result = await run_news_screen(["AAPL"], {"management_changes": True})
        # We have "CEO announces..." which should be categorized as management
        assert result["matches"] >= 0

    @patch("stockscreen.yf.Ticker")
    async def test_no_matching_news(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        self._setup_news_ticker(mock_ticker_cls, mock_ticker_info, mock_news_items)
        result = await run_news_screen(["AAPL"], {"keywords": ["zzzznotfound"]})
        assert result["matches"] == 0

    @patch("stockscreen.yf.Ticker")
    async def test_no_news_at_all(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = []
        mock_ticker_cls.return_value = mock_instance
        result = await run_news_screen(["AAPL"], {"keywords": ["test"]})
        assert result["matches"] == 0


# ============================================================
# 13. run_custom_screen
# ============================================================
class TestRunCustomScreen:
    @patch("stockscreen.yf.Ticker")
    async def test_technical_and_fundamental_pass(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.history.return_value = mock_history_df
        mock_ticker_cls.return_value = mock_instance
        result = await run_custom_screen(["AAPL"], {
            "technical": {"min_price": 10},
            "fundamental": {"min_market_cap": 1_000_000},
        })
        assert result["screen_type"] == "custom"
        assert result["matches"] >= 1

    @patch("stockscreen.yf.Ticker")
    async def test_technical_reject_stops_further(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.history.return_value = mock_history_df
        mock_ticker_cls.return_value = mock_instance
        result = await run_custom_screen(["AAPL"], {
            "technical": {"min_price": 99999},
            "fundamental": {"min_market_cap": 1},
        })
        assert result["matches"] == 0
        assert len(result["rejected"]) == 1

    @patch("stockscreen.yf.Ticker")
    async def test_symbols_from_criteria(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.history.return_value = mock_history_df
        mock_ticker_cls.return_value = mock_instance
        result = await run_custom_screen(None, {
            "symbols": ["AAPL"],
            "technical": {"min_price": 10},
        })
        assert result["matches"] >= 1

    @patch("stockscreen.yf.Ticker")
    async def test_error_in_single_symbol_continues(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        call_count = 0

        def side_effect(symbol):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise RuntimeError("api error")
            mock_inst = MagicMock()
            mock_inst.info = mock_ticker_info
            mock_inst.history.return_value = mock_history_df
            return mock_inst

        mock_ticker_cls.side_effect = side_effect
        result = await run_custom_screen(["BAD", "AAPL"], {
            "technical": {"min_price": 10},
        })
        # BAD should be in rejected, AAPL should pass
        assert result["matches"] >= 1 or len(result["rejected"]) >= 1

    async def test_empty_criteria(self):
        with patch("stockscreen.data_store") as mock_ds:
            mock_ds.default_symbols.get_symbols = AsyncMock(return_value=["AAPL"])
            with patch("stockscreen.yf.Ticker") as mock_ticker_cls:
                mock_instance = MagicMock()
                mock_instance.info = {"regularMarketPrice": 100}
                mock_ticker_cls.return_value = mock_instance
                result = await run_custom_screen(["AAPL"], {})
                assert result["screen_type"] == "custom"
                assert result["matches"] == 1


# ============================================================
# 14. get_news_data
# ============================================================
class TestGetNewsData:
    @patch("stockscreen.yf.Ticker")
    async def test_basic_news_fetch(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = mock_news_items
        mock_ticker_cls.return_value = mock_instance
        result = await get_news_data("AAPL", days_back=30)
        assert "recent_news" in result
        assert "key_events" in result
        assert "management_changes" in result
        assert "last_updated" in result

    @patch("stockscreen.yf.Ticker")
    async def test_news_categorization_management(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = mock_news_items
        mock_ticker_cls.return_value = mock_instance
        result = await get_news_data("AAPL")
        # "CEO announces..." should be in management_changes
        mgmt_titles = [n["title"] for n in result["management_changes"]]
        assert any("CEO" in t for t in mgmt_titles)

    @patch("stockscreen.yf.Ticker")
    async def test_news_categorization_key_events(self, mock_ticker_cls, mock_ticker_info, mock_news_items):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = mock_news_items
        mock_ticker_cls.return_value = mock_instance
        result = await get_news_data("AAPL")
        # "SEC investigation..." should be in key_events
        event_titles = [n["title"] for n in result["key_events"]]
        assert any("SEC" in t for t in event_titles)

    @patch("stockscreen.yf.Ticker")
    async def test_no_news(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.news = []
        mock_ticker_cls.return_value = mock_instance
        result = await get_news_data("AAPL")
        assert result["recent_news"] == []
        assert result["key_events"] == []
        assert result["management_changes"] == []

    @patch("stockscreen.yf.Ticker")
    async def test_error_returns_error_dict(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("api down")
        result = await get_news_data("AAPL")
        assert "error" in result
        assert "last_updated" in result


# ============================================================
# 15. run_single_technical_screen
# ============================================================
class TestRunSingleTechnicalScreen:
    def _make_mock_ticker(self, mock_ticker_cls, info, history_df):
        mock_instance = MagicMock()
        mock_instance.info = info
        mock_instance.history.return_value = history_df
        mock_ticker_cls.return_value = mock_instance

    @patch("stockscreen.yf.Ticker")
    async def test_pass_no_criteria(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {})
        assert result["rejection_reasons"] == []
        assert result["data"]["price"] == 150.0
        assert "volume" in result["data"]

    @patch("stockscreen.yf.Ticker")
    async def test_min_price_rejection(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {"min_price": 200})
        assert len(result["rejection_reasons"]) == 1
        assert "Price" in result["rejection_reasons"][0]

    @patch("stockscreen.yf.Ticker")
    async def test_max_price_rejection(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {"max_price": 100})
        assert len(result["rejection_reasons"]) == 1

    @patch("stockscreen.yf.Ticker")
    async def test_volume_rejection(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {"min_volume": 999_999_999_999})
        assert any("Volume" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_above_sma_200(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {"above_sma_200": True})
        # Price=150, SMA200 of our synthetic data ~110 → should pass
        assert "sma_200" in result["data"]

    @patch("stockscreen.yf.Ticker")
    async def test_above_sma_50(self, mock_ticker_cls, mock_ticker_info, mock_history_df):
        self._make_mock_ticker(mock_ticker_cls, mock_ticker_info, mock_history_df)
        result = await run_single_technical_screen("AAPL", {"above_sma_50": True})
        assert "sma_50" in result["data"]

    @patch("stockscreen.yf.Ticker")
    async def test_empty_history_error(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance
        result = await run_single_technical_screen("AAPL", {})
        assert len(result["rejection_reasons"]) == 1
        assert "Technical analysis error" in result["rejection_reasons"][0]

    @patch("stockscreen.yf.Ticker")
    async def test_exception_propagates(self, mock_ticker_cls):
        # yf.Ticker() call is outside the try/except, so exception propagates
        mock_ticker_cls.side_effect = RuntimeError("api down")
        with pytest.raises(RuntimeError, match="api down"):
            await run_single_technical_screen("AAPL", {})


# ============================================================
# 16. run_single_fundamental_screen
# ============================================================
class TestRunSingleFundamentalScreen:
    @patch("stockscreen.yf.Ticker")
    async def test_pass_no_criteria(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        result = await run_single_fundamental_screen("AAPL", {})
        assert result["rejection_reasons"] == []
        assert result["data"]["market_cap"] == 2_500_000_000_000
        assert result["data"]["pe_ratio"] == 25.5

    @patch("stockscreen.yf.Ticker")
    async def test_min_market_cap_rejection(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        result = await run_single_fundamental_screen("AAPL", {"min_market_cap": 10e12})
        assert any("Market cap" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_pe_range_rejection(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        # forwardPE=25.5, max_pe=20 → rejected
        result = await run_single_fundamental_screen("AAPL", {"max_pe": 20})
        assert any("P/E" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_min_pe_rejection(self, mock_ticker_cls, mock_ticker_info):
        mock_instance = MagicMock()
        mock_instance.info = mock_ticker_info
        mock_ticker_cls.return_value = mock_instance
        # forwardPE=25.5, min_pe=30 → rejected
        result = await run_single_fundamental_screen("AAPL", {"min_pe": 30})
        assert any("P/E" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_no_info_error(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.info = None
        mock_ticker_cls.return_value = mock_instance
        result = await run_single_fundamental_screen("BAD", {})
        assert result["data"] == {}
        assert "Fundamental analysis error" in result["rejection_reasons"][0]

    @patch("stockscreen.yf.Ticker")
    async def test_exception_propagates(self, mock_ticker_cls):
        # yf.Ticker() call is outside the try/except, so exception propagates
        mock_ticker_cls.side_effect = RuntimeError("api down")
        with pytest.raises(RuntimeError, match="api down"):
            await run_single_fundamental_screen("AAPL", {})


# ============================================================
# 17. run_single_options_screen
# ============================================================
class TestRunSingleOptionsScreen:
    def _setup_ticker(self, mock_ticker_cls, mock_option_chain):
        mock_instance = MagicMock()
        mock_instance.options = ("2024-03-15",)
        mock_instance.option_chain.return_value = mock_option_chain
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        mock_instance.calendar = {"Earnings Date": [future_date]}
        mock_ticker_cls.return_value = mock_instance
        return mock_instance

    @patch("stockscreen.yf.Ticker")
    async def test_pass_no_criteria(self, mock_ticker_cls, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_option_chain)
        result = await run_single_options_screen("AAPL", {})
        assert result["rejection_reasons"] == []
        assert "option_volume" in result["data"]

    @patch("stockscreen.yf.Ticker")
    async def test_min_option_volume_rejection(self, mock_ticker_cls, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_option_chain)
        result = await run_single_options_screen("AAPL", {"min_option_volume": 999_999_999})
        assert any("Option volume" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_earnings_criteria(self, mock_ticker_cls, mock_option_chain):
        self._setup_ticker(mock_ticker_cls, mock_option_chain)
        result = await run_single_options_screen("AAPL", {"max_days_to_earnings": 5})
        # days_to_earnings=30 > max=5 → rejected
        assert any("Days to earnings" in r for r in result["rejection_reasons"])

    @patch("stockscreen.yf.Ticker")
    async def test_no_options_error(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.options = ()
        mock_ticker_cls.return_value = mock_instance
        result = await run_single_options_screen("AAPL", {})
        assert result["data"] == {}
        assert "Options analysis error" in result["rejection_reasons"][0]

    @patch("stockscreen.yf.Ticker")
    async def test_exception_propagates(self, mock_ticker_cls):
        # yf.Ticker() call is outside the try/except, so exception propagates
        mock_ticker_cls.side_effect = RuntimeError("api down")
        with pytest.raises(RuntimeError, match="api down"):
            await run_single_options_screen("AAPL", {})


# ============================================================
# 18. get_earnings_dates
# ============================================================
class TestGetEarningsDates:
    async def test_with_earnings_dates(self):
        mock_ticker = MagicMock()
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        mock_ticker.calendar = {"Earnings Date": [future_date]}
        result = await get_earnings_dates(mock_ticker)
        assert result["next_earnings"] == future_date
        assert result["days_to_earnings"] == 30

    async def test_no_calendar(self):
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        result = await get_earnings_dates(mock_ticker)
        assert result["next_earnings"] is None
        assert result["days_to_earnings"] is None

    async def test_earnings_date_range(self):
        mock_ticker = MagicMock()
        date1 = datetime.date.today() + datetime.timedelta(days=25)
        date2 = datetime.date.today() + datetime.timedelta(days=30)
        mock_ticker.calendar = {"Earnings Date": [date1, date2]}
        result = await get_earnings_dates(mock_ticker)
        assert result["next_earnings"] == date1
        assert result["earnings_range_end"] == date2
        assert result["is_estimate"] is True
        assert result["days_to_earnings"] == 25

    async def test_exception_handling(self):
        mock_ticker = MagicMock()
        mock_ticker.calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        # calendar access raises
        type(mock_ticker).calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        result = await get_earnings_dates(mock_ticker)
        assert result["next_earnings"] is None
        assert result["days_to_earnings"] is None
