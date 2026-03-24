"""Tests for MarketDataFacade — TDD step 2."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from stockscreen.exceptions import APIError
from stockscreen.providers.boursorama import BoursoramaQuote
from stockscreen.providers.euronext import EuronextRecord
from stockscreen.providers.facade import MarketDataFacade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_euronext_record(
    isin="FR0000131104",
    symbol="TTE",
    mic="XPAR",
) -> EuronextRecord:
    suffix = {"XPAR": ".PA", "XETR": ".DE"}.get(mic, "")
    return EuronextRecord(
        isin=isin,
        symbol=symbol,
        name="TotalEnergies SE",
        mic=mic,
        yahoo_ticker=f"{symbol}{suffix}",
        cached_at="2026-01-01T00:00:00",
    )


def _make_bourso_quote(
    dividende=3.02,
    rendement=5.08,
    last_dividend_date="2026-03-18",
    consensus="Acheter",
) -> BoursoramaQuote:
    return BoursoramaQuote(
        isin="FR0000131104",
        code_bourso="1rTTE",
        nom="TotalEnergies SE",
        lien="https://www.boursorama.com/cours/1rTTE/",
        cours=59.42,
        dividende=dividende,
        rendement=rendement,
        last_dividend_date=last_dividend_date,
        consensus=consensus,
        performance=[{"annee": "2023", "ca": 218.9, "rn": 19.6, "marge": 8.95}],
        cached_at="2026-01-01T00:00:00",
    )


YAHOO_INFO = {
    "regularMarketPrice": 59.42,
    "marketCap": 140_000_000_000,
    "trailingPE": 9.5,
    "dividendRate": 3.02,
    "dividendYield": 0.0508,
    "trailingAnnualDividendRate": 3.02,
    "volume": 8_000_000,
    "symbol": "TTE.PA",
}


def _make_facade(yahoo=None, boursorama=None, euronext=None) -> MarketDataFacade:
    yahoo = yahoo or AsyncMock()
    boursorama = boursorama or AsyncMock()
    euronext = euronext or AsyncMock()
    return MarketDataFacade(yahoo=yahoo, boursorama=boursorama, euronext=euronext)


# ---------------------------------------------------------------------------
# 1. Résolution d'identifiant
# ---------------------------------------------------------------------------

class TestIdentifierResolution:
    async def test_yahoo_ticker_resolved_directly(self):
        """Ticker Yahoo (ex: TTE.PA) → Yahoo reçoit TTE.PA, Boursorama reçoit TTE."""
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        await facade.get_quote("TTE.PA")

        yahoo.get_ticker_info.assert_called_once_with("TTE.PA")
        # Boursorama reçoit le ticker court sans suffixe
        bourso.get_quote.assert_called_once_with("TTE")

    async def test_isin_resolved_via_euronext(self):
        """ISIN → EuronextProvider.resolve_ticker → yahoo_ticker pour Yahoo."""
        euronext = AsyncMock()
        euronext.resolve_ticker.return_value = _make_euronext_record()
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso, euronext=euronext)
        await facade.get_quote("FR0000131104")

        euronext.resolve_ticker.assert_called_once_with("FR0000131104")
        yahoo.get_ticker_info.assert_called_once_with("TTE.PA")

    async def test_isin_sends_isin_to_boursorama(self):
        """Pour un ISIN, Boursorama reçoit l'ISIN (plus fiable qu'un ticker court)."""
        euronext = AsyncMock()
        euronext.resolve_ticker.return_value = _make_euronext_record()
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso, euronext=euronext)
        await facade.get_quote("FR0000131104")

        bourso.get_quote.assert_called_once_with("FR0000131104")

    async def test_unresolvable_isin_raises_api_error(self):
        """ISIN non résolvable par Euronext → APIError."""
        euronext = AsyncMock()
        euronext.resolve_ticker.return_value = None
        yahoo = AsyncMock()

        facade = _make_facade(yahoo=yahoo, euronext=euronext)
        with pytest.raises(APIError):
            await facade.get_quote("XX9999999999")

    async def test_euronext_not_called_for_ticker(self):
        """Pour un ticker Yahoo, EuronextProvider n'est pas sollicité."""
        euronext = AsyncMock()
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso, euronext=euronext)
        await facade.get_quote("TTE.PA")

        euronext.resolve_ticker.assert_not_called()

    async def test_euronext_called_once_not_twice(self):
        """EuronextProvider.resolve_ticker appelé une seule fois même si Boursorama échoue."""
        euronext = AsyncMock()
        euronext.resolve_ticker.return_value = _make_euronext_record()
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.side_effect = APIError("not found")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso, euronext=euronext)
        await facade.get_quote("FR0000131104")

        assert euronext.resolve_ticker.call_count == 1


# ---------------------------------------------------------------------------
# 2. Stratégie dividende — Boursorama-first
# ---------------------------------------------------------------------------

class TestDividendStrategy:
    async def test_dividend_from_boursorama_when_available(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = {**YAHOO_INFO, "dividendRate": 1.0}
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote(dividende=3.02, rendement=5.08)

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["dividende"] == 3.02
        assert result["rendement"] == pytest.approx(5.08)

    async def test_last_dividend_date_from_boursorama(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote(last_dividend_date="2026-03-18")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["last_dividend_date"] == "2026-03-18"

    async def test_dividend_fallback_to_yahoo_on_api_error(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = {**YAHOO_INFO, "dividendRate": 3.02, "regularMarketPrice": 59.42}
        bourso = AsyncMock()
        bourso.get_quote.side_effect = APIError("scrape failed")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        # rendement calculé depuis Yahoo
        assert result["rendement"] == pytest.approx(3.02 / 59.42 * 100, rel=1e-3)

    async def test_dividend_fallback_to_yahoo_when_boursorama_none(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = {**YAHOO_INFO, "dividendRate": 3.02, "regularMarketPrice": 59.42}
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote(dividende=None, rendement=None)

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["rendement"] == pytest.approx(3.02 / 59.42 * 100, rel=1e-3)

    async def test_consensus_from_boursorama(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote(consensus="Acheter")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["consensus"] == "Acheter"

    async def test_consensus_absent_when_boursorama_fails(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.side_effect = APIError("scrape failed")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result.get("consensus") is None

    async def test_performance_from_boursorama(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert len(result["performance"]) == 1
        assert result["performance"][0]["annee"] == "2023"

    async def test_performance_empty_list_when_boursorama_fails(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.side_effect = APIError("scrape failed")

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["performance"] == []


# ---------------------------------------------------------------------------
# 3. Données techniques — toujours depuis Yahoo
# ---------------------------------------------------------------------------

class TestYahooFields:
    async def test_market_cap_from_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = {**YAHOO_INFO, "marketCap": 140_000_000_000}
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["marketCap"] == 140_000_000_000

    async def test_price_from_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = {**YAHOO_INFO, "regularMarketPrice": 59.42}
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert result["regularMarketPrice"] == 59.42

    async def test_result_is_flat_dict(self):
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_quote("TTE.PA")

        assert isinstance(result, dict)
        # Pas de dict imbriqué pour les données de base
        assert "marketCap" in result
        assert "dividende" in result


# ---------------------------------------------------------------------------
# 4. Appels parallèles
# ---------------------------------------------------------------------------

class TestParallelCalls:
    async def test_yahoo_and_boursorama_called_in_parallel(self):
        """Les deux appels doivent être lancés (asyncio.gather)."""
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        await facade.get_quote("TTE.PA")

        yahoo.get_ticker_info.assert_called_once()
        bourso.get_quote.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Méthodes déléguées à Yahoo
# ---------------------------------------------------------------------------

class TestDelegatedMethods:
    async def test_get_history_delegates_to_yahoo(self):
        yahoo = AsyncMock()
        expected_df = pd.DataFrame({"Close": [59.0, 60.0]})
        yahoo.get_history.return_value = expected_df

        facade = _make_facade(yahoo=yahoo)
        result = await facade.get_history("TTE.PA", period="1y")

        yahoo.get_history.assert_called_once_with("TTE.PA", period="1y")
        assert result is expected_df

    async def test_get_news_delegates_to_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_news.return_value = [{"title": "TTE news"}]

        facade = _make_facade(yahoo=yahoo)
        result = await facade.get_news("TTE.PA")

        yahoo.get_news.assert_called_once_with("TTE.PA")
        assert result == [{"title": "TTE news"}]

    async def test_get_option_chain_delegates_to_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_option_chain.return_value = MagicMock()

        facade = _make_facade(yahoo=yahoo)
        await facade.get_option_chain("TTE.PA", "2026-06-20")

        yahoo.get_option_chain.assert_called_once_with("TTE.PA", "2026-06-20")

    async def test_get_option_expirations_delegates_to_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_option_expirations.return_value = ("2026-06-20", "2026-09-19")

        facade = _make_facade(yahoo=yahoo)
        result = await facade.get_option_expirations("TTE.PA")

        yahoo.get_option_expirations.assert_called_once_with("TTE.PA")
        assert "2026-06-20" in result

    async def test_get_earnings_dates_delegates_to_yahoo(self):
        yahoo = AsyncMock()
        yahoo.get_earnings_dates.return_value = {"next_earnings": None, "days_to_earnings": None}

        facade = _make_facade(yahoo=yahoo)
        result = await facade.get_earnings_dates("TTE.PA")

        yahoo.get_earnings_dates.assert_called_once_with("TTE.PA")
        assert "next_earnings" in result

    async def test_get_ticker_info_equivalent_to_get_quote(self):
        """get_ticker_info est l'alias utilisé par ScreenerService — même résultat que get_quote."""
        yahoo = AsyncMock()
        yahoo.get_ticker_info.return_value = YAHOO_INFO
        bourso = AsyncMock()
        bourso.get_quote.return_value = _make_bourso_quote()

        facade = _make_facade(yahoo=yahoo, boursorama=bourso)
        result = await facade.get_ticker_info("TTE.PA")

        assert isinstance(result, dict)
        assert "dividende" in result
        assert "marketCap" in result


# ---------------------------------------------------------------------------
# 6. MarketDataFacade n'importe pas yfinance
# ---------------------------------------------------------------------------

class TestNoDirectYfinanceImport:
    def test_facade_module_does_not_import_yfinance(self):
        import importlib
        import stockscreen.providers.facade as facade_module
        assert not hasattr(facade_module, "yf"), "facade should not import yfinance directly"
