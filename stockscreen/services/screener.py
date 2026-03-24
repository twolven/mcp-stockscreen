"""Screener service — unified stock screening (technical, fundamental, options, news, custom)."""

import datetime
import logging

import pandas as pd

from stockscreen.providers.yahoo import YahooProvider
from stockscreen.services.news import NewsService
from stockscreen.store.data_store import ScreenerDataStore

logger = logging.getLogger("stockscreen-server-v1")

_SCREEN_TYPES = {"technical", "fundamental", "options", "news", "custom"}


def _dividend_yield_pct(info: dict) -> float:
    """Return the dividend yield as a plain percentage (e.g. 4.5 for 4.5%).

    Yahoo Finance is inconsistent:
    - US equities:        dividendYield = 0.045  (decimal)
    - Non-US / some ETFs: dividendYield = 4.5    (already a %)
    - Stale / missing:    dividendYield = None

    Strategy (in order of preference):
    1. Compute from dividendRate (annual $/€ per share) ÷ current price — format-agnostic.
    2. Fall back to dividendYield: if the raw value > 1.0 it is already a percentage,
       otherwise multiply by 100.
    """
    price = info.get("regularMarketPrice") or info.get("currentPrice") or 0
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0
    if rate and price:
        return rate / price * 100

    raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
    if raw > 1.0:
        return float(raw)   # already a percentage
    return raw * 100


class ScreenerService:
    """Unified screening service with injection of provider, store, and news service."""

    def __init__(
        self,
        provider: YahooProvider,
        store: ScreenerDataStore,
        news_service: NewsService,
        symbol_service=None,   # SymbolService | None — optional to stay backwards-compatible
    ):
        self.provider = provider
        self.store = store
        self.news_service = news_service
        self.symbol_service = symbol_service

    async def run(
        self,
        screen_type: str,
        criteria: dict,
        symbols: list[str] | None = None,
        watchlist_name: str | None = None,
    ) -> dict:
        """Dispatch to the appropriate screen type.

        Symbol resolution priority:
          1. ``symbols`` parameter
          2. ``watchlist_name`` → loaded from store
          3. ``criteria["symbols"]`` key
          4. ``criteria["category"]`` → resolved via SymbolService
          5. Empty list
        """
        if screen_type not in _SCREEN_TYPES:
            raise ValueError(f"Invalid screen type: {screen_type}")

        if symbols is None:
            if watchlist_name:
                symbols = self.store.load_watchlist(watchlist_name) or []
            elif "symbols" in criteria:
                raw = criteria["symbols"]
                symbols = [raw] if isinstance(raw, str) else list(raw)
            elif "category" in criteria and self.symbol_service:
                symbols = await self.symbol_service.get(criteria["category"])
            else:
                symbols = []

        if screen_type == "technical":
            return await self._run_technical(symbols, criteria)
        if screen_type == "fundamental":
            return await self._run_fundamental(symbols, criteria)
        if screen_type == "options":
            return await self._run_options(symbols, criteria)
        if screen_type == "news":
            return await self.news_service.screen_news(symbols, criteria)
        # custom
        return await self._run_custom(symbols, criteria)

    # ------------------------------------------------------------------
    # Technical screen
    # ------------------------------------------------------------------

    async def _run_technical(self, symbols: list[str], criteria: dict) -> dict:
        results = []
        rejected = []

        for symbol in symbols:
            try:
                info = await self.provider.get_ticker_info(symbol)
                hist = await self.provider.get_history(symbol)

                if hist.empty:
                    rejected.append({"symbol": symbol, "error": "No historical data available"})
                    continue

                rejection_reasons = []
                data = hist.copy()

                # Current price
                try:
                    current_price = (
                        (info or {}).get("regularMarketPrice")
                        or (info or {}).get("currentPrice")
                        or data["Close"].iloc[-1]
                    )
                    if pd.isna(current_price):
                        raise ValueError("No valid price data")
                except Exception as e:
                    rejected.append({"symbol": symbol, "error": f"Price data error: {e}"})
                    continue

                # Price criteria
                if "min_price" in criteria and current_price < criteria["min_price"]:
                    rejection_reasons.append(
                        f"Price ({current_price:.2f}) < minimum ({criteria['min_price']})"
                    )
                if "max_price" in criteria and current_price > criteria["max_price"]:
                    rejection_reasons.append(
                        f"Price ({current_price:.2f}) > maximum ({criteria['max_price']})"
                    )

                # Volume
                avg_volume = None
                if "Volume" in data.columns:
                    v = data["Volume"].mean()
                    avg_volume = None if pd.isna(v) else v
                if avg_volume is not None and "min_volume" in criteria and avg_volume < criteria["min_volume"]:
                    rejection_reasons.append(
                        f"Volume ({avg_volume:.0f}) < minimum ({criteria['min_volume']})"
                    )

                # Moving averages
                data["SMA_20"] = data["Close"].rolling(window=20).mean()
                data["SMA_50"] = data["Close"].rolling(window=50).mean()
                data["SMA_200"] = data["Close"].rolling(window=200).mean()
                sma_20 = data["SMA_20"].iloc[-1]
                sma_50 = data["SMA_50"].iloc[-1]
                sma_200 = data["SMA_200"].iloc[-1]

                if criteria.get("above_sma_200", False):
                    if pd.isna(sma_200):
                        rejection_reasons.append("Insufficient data for SMA200 calculation")
                    elif current_price <= sma_200:
                        rejection_reasons.append(
                            f"Price ({current_price:.2f}) below SMA200 ({sma_200:.2f})"
                        )

                if criteria.get("above_sma_50", False):
                    if pd.isna(sma_50):
                        rejection_reasons.append("Insufficient data for SMA50 calculation")
                    elif current_price <= sma_50:
                        rejection_reasons.append(
                            f"Price ({current_price:.2f}) below SMA50 ({sma_50:.2f})"
                        )

                # RSI
                current_rsi = None
                try:
                    delta = data["Close"].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14, min_periods=1).mean()
                    avg_loss = loss.rolling(window=14, min_periods=1).mean()
                    rs = avg_gain / avg_loss
                    data["RSI"] = 100 - (100 / (1 + rs))
                    rsi_val = data["RSI"].iloc[-1]
                    if not pd.isna(rsi_val):
                        current_rsi = rsi_val
                        if "min_rsi" in criteria and current_rsi < criteria["min_rsi"]:
                            rejection_reasons.append(
                                f"RSI ({current_rsi:.1f}) < minimum ({criteria['min_rsi']})"
                            )
                        if "max_rsi" in criteria and current_rsi > criteria["max_rsi"]:
                            rejection_reasons.append(
                                f"RSI ({current_rsi:.1f}) > maximum ({criteria['max_rsi']})"
                            )
                except Exception as e:
                    rejection_reasons.append(f"RSI calculation error: {e}")

                # ATR
                current_atr = None
                try:
                    high_low = data["High"] - data["Low"]
                    high_close = abs(data["High"] - data["Close"].shift())
                    low_close = abs(data["Low"] - data["Close"].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    data["ATR"] = tr.rolling(window=14).mean()
                    atr_val = data["ATR"].iloc[-1]
                    if not pd.isna(atr_val):
                        current_atr = atr_val
                        if "max_atr_pct" in criteria:
                            atr_pct = (current_atr / current_price) * 100
                            if atr_pct > criteria["max_atr_pct"]:
                                rejection_reasons.append(
                                    f"ATR% ({atr_pct:.1f}%) > maximum ({criteria['max_atr_pct']}%)"
                                )
                except Exception as e:
                    rejection_reasons.append(f"ATR calculation error: {e}")

                # Price changes
                changes = {"1d": None, "5d": None, "20d": None}
                try:
                    for key, periods in [("1d", 1), ("5d", 5), ("20d", 20)]:
                        val = data["Close"].pct_change(periods=periods).iloc[-1]
                        changes[key] = None if pd.isna(val) else val * 100
                except Exception:
                    pass

                def _safe(v):
                    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

                entry = {
                    "symbol": symbol,
                    "price": current_price,
                    "volume": avg_volume,
                    "rsi": current_rsi,
                    "sma_20": _safe(sma_20),
                    "sma_50": _safe(sma_50),
                    "sma_200": _safe(sma_200),
                    "atr": current_atr,
                    "price_changes": changes,
                }

                if rejection_reasons:
                    entry["rejection_reasons"] = rejection_reasons
                    rejected.append(entry)
                else:
                    entry["atr_pct"] = (current_atr / current_price * 100) if current_atr else None
                    entry["ma_distances"] = {
                        "pct_from_20sma": (
                            ((current_price / _safe(sma_20)) - 1) * 100 if _safe(sma_20) else None
                        ),
                        "pct_from_50sma": (
                            ((current_price / _safe(sma_50)) - 1) * 100 if _safe(sma_50) else None
                        ),
                        "pct_from_200sma": (
                            ((current_price / _safe(sma_200)) - 1) * 100 if _safe(sma_200) else None
                        ),
                    }
                    results.append(entry)

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                rejected.append({"symbol": symbol, "error": str(e)})

        return {
            "screen_type": "technical",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Fundamental screen
    # ------------------------------------------------------------------

    async def _run_fundamental(self, symbols: list[str], criteria: dict) -> dict:
        results = []
        rejected = []

        for symbol in symbols:
            try:
                info = await self.provider.get_ticker_info(symbol)
                if not info:
                    rejected.append({"symbol": symbol, "error": "No fundamental data available"})
                    continue

                is_etf = info.get("quoteType") == "ETF"

                if is_etf:
                    current_price = info.get("regularMarketPrice") or info.get("previousClose")
                    volume = info.get("regularMarketVolume") or info.get("averageVolume")
                    aum = info.get("totalAssets", 0)
                    expense_ratio = info.get("expenseRatio", 0)

                    if "min_aum" in criteria and aum < criteria["min_aum"]:
                        rejected.append({"symbol": symbol, "rejection_reasons": ["AUM < minimum"]})
                        continue
                    if "max_expense_ratio" in criteria and expense_ratio > criteria["max_expense_ratio"]:
                        rejected.append(
                            {"symbol": symbol, "rejection_reasons": ["Expense ratio > maximum"]}
                        )
                        continue
                    if "min_volume" in criteria and (not volume or volume < criteria["min_volume"]):
                        rejected.append({"symbol": symbol, "rejection_reasons": ["Volume < minimum"]})
                        continue

                    results.append(
                        {
                            "symbol": symbol,
                            "price": current_price,
                            "aum": aum,
                            "expense_ratio": expense_ratio,
                            "average_volume": volume,
                            "category": info.get("category", "Unknown"),
                            "asset_class": info.get("assetClass", "Unknown"),
                        }
                    )
                    continue

                # Regular stock
                rejection_reasons = []

                market_cap = info.get("marketCap", 0)
                if "min_market_cap" in criteria and market_cap < criteria["min_market_cap"]:
                    rejection_reasons.append(
                        f"Market cap ({market_cap}) < minimum ({criteria['min_market_cap']})"
                    )

                pe_ratio = info.get("forwardPE", 0)
                if "min_pe" in criteria and pe_ratio < criteria["min_pe"]:
                    rejection_reasons.append(f"P/E ({pe_ratio}) < minimum ({criteria['min_pe']})")
                if "max_pe" in criteria and pe_ratio > criteria["max_pe"]:
                    rejection_reasons.append(f"P/E ({pe_ratio}) > maximum ({criteria['max_pe']})")

                dividend_yield = _dividend_yield_pct(info)
                if "min_dividend" in criteria and dividend_yield < criteria["min_dividend"]:
                    rejection_reasons.append(
                        f"Dividend yield ({dividend_yield:.2f}%) < minimum ({criteria['min_dividend']}%)"
                    )

                revenue_growth = info.get("revenueGrowth") or 0
                if "min_revenue_growth" in criteria and revenue_growth < criteria["min_revenue_growth"]:
                    rejection_reasons.append(
                        f"Revenue growth ({revenue_growth:.2f}) < minimum ({criteria['min_revenue_growth']})"
                    )

                if rejection_reasons:
                    rejected.append({"symbol": symbol, "rejection_reasons": rejection_reasons})
                else:
                    results.append(
                        {
                            "symbol": symbol,
                            "market_cap": market_cap,
                            "pe_ratio": pe_ratio,
                            "dividend_yield": dividend_yield,
                            "revenue_growth": revenue_growth,
                            "sector": info.get("sector", "Unknown"),
                            "industry": info.get("industry", "Unknown"),
                        }
                    )

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                rejected.append({"symbol": symbol, "error": str(e)})

        return {
            "screen_type": "fundamental",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Options screen
    # ------------------------------------------------------------------

    async def _run_options(self, symbols: list[str], criteria: dict) -> dict:
        results = []
        rejected = []

        for symbol in symbols:
            try:
                info = await self.provider.get_ticker_info(symbol)
                earnings_info = await self.provider.get_earnings_dates(symbol)
                rejection_reasons = []

                # Earnings date criteria (checked before fetching chain)
                if "min_days_to_earnings" in criteria:
                    days = earnings_info.get("days_to_earnings")
                    if days is None or days < criteria["min_days_to_earnings"]:
                        rejection_reasons.append(
                            f"Days to earnings ({days}) < minimum ({criteria['min_days_to_earnings']})"
                        )
                if "max_days_to_earnings" in criteria:
                    days = earnings_info.get("days_to_earnings")
                    if days is None or days > criteria["max_days_to_earnings"]:
                        rejection_reasons.append(
                            f"Days to earnings ({days}) > maximum ({criteria['max_days_to_earnings']})"
                        )

                if rejection_reasons:
                    rejected.append(
                        {
                            "symbol": symbol,
                            "rejection_reasons": rejection_reasons,
                            "next_earnings": earnings_info.get("next_earnings"),
                            "days_to_earnings": earnings_info.get("days_to_earnings"),
                        }
                    )
                    continue

                # Options expiration dates
                exp_dates = await self.provider.get_option_expirations(symbol)
                if not exp_dates:
                    rejected.append(
                        {"symbol": symbol, "rejection_reasons": ["No options expiration dates available"]}
                    )
                    continue

                # Current price
                current_price = (info or {}).get("regularMarketPrice") or (info or {}).get("currentPrice")
                if not current_price or pd.isna(current_price):
                    try:
                        hist = await self.provider.get_history(symbol, period="1d")
                        current_price = hist["Close"].iloc[-1] if not hist.empty else None
                    except Exception:
                        current_price = None

                # Option chain analysis
                try:
                    chain = await self.provider.get_option_chain(symbol, exp_dates[0])
                    if not chain or not hasattr(chain, "calls") or not hasattr(chain, "puts"):
                        raise ValueError("Invalid options chain data")

                    calls = chain.calls
                    puts = chain.puts

                    total_volume = int(calls["volume"].sum() + puts["volume"].sum())
                    total_oi = int(calls["openInterest"].sum() + puts["openInterest"].sum())

                    if "min_option_volume" in criteria and total_volume < criteria["min_option_volume"]:
                        rejection_reasons.append(
                            f"Option volume ({total_volume}) < minimum ({criteria['min_option_volume']})"
                        )

                    put_volume = puts["volume"].sum()
                    call_volume = calls["volume"].sum()
                    put_call_ratio = put_volume / max(1, call_volume)

                    if "min_put_call_ratio" in criteria and put_call_ratio < criteria["min_put_call_ratio"]:
                        rejection_reasons.append(
                            f"Put/Call ratio ({put_call_ratio:.2f}) < minimum ({criteria['min_put_call_ratio']})"
                        )

                    # ATM metrics (only when price is known)
                    atm_strike = avg_iv = atm_call_iv = atm_put_iv = avg_spread = None
                    atm_call_spread = atm_put_spread = None
                    if current_price is not None and not calls.empty:
                        atm_strike = min(calls["strike"], key=lambda x: abs(x - current_price))
                        atm_calls = calls[calls["strike"] == atm_strike]
                        atm_puts = puts[puts["strike"] == atm_strike]

                        if not atm_calls.empty and not atm_puts.empty:
                            atm_call_iv = float(atm_calls["impliedVolatility"].iloc[0]) * 100
                            atm_put_iv = float(atm_puts["impliedVolatility"].iloc[0]) * 100
                            avg_iv = (atm_call_iv + atm_put_iv) / 2

                            if "min_iv" in criteria and avg_iv < criteria["min_iv"]:
                                rejection_reasons.append(
                                    f"IV ({avg_iv:.1f}%) < minimum ({criteria['min_iv']}%)"
                                )
                            if "max_iv" in criteria and avg_iv > criteria["max_iv"]:
                                rejection_reasons.append(
                                    f"IV ({avg_iv:.1f}%) > maximum ({criteria['max_iv']}%)"
                                )

                            call_ask = float(atm_calls["ask"].iloc[0])
                            put_ask = float(atm_puts["ask"].iloc[0])
                            atm_call_spread = (
                                (call_ask - float(atm_calls["bid"].iloc[0])) / call_ask * 100
                                if call_ask else 0
                            )
                            atm_put_spread = (
                                (put_ask - float(atm_puts["bid"].iloc[0])) / put_ask * 100
                                if put_ask else 0
                            )
                            avg_spread = (atm_call_spread + atm_put_spread) / 2

                            if "max_spread" in criteria and avg_spread > criteria["max_spread"]:
                                rejection_reasons.append(
                                    f"Avg spread ({avg_spread:.1f}%) > maximum ({criteria['max_spread']}%)"
                                )

                    if rejection_reasons:
                        rejected.append(
                            {
                                "symbol": symbol,
                                "price": current_price,
                                "rejection_reasons": rejection_reasons,
                                "implied_volatility": avg_iv / 100 if avg_iv is not None else None,
                                "implied_volatility_pct": avg_iv,
                                "option_volume": total_volume,
                                "put_call_ratio": put_call_ratio,
                                "next_earnings": earnings_info.get("next_earnings"),
                                "days_to_earnings": earnings_info.get("days_to_earnings"),
                            }
                        )
                    else:
                        entry: dict = {
                            "symbol": symbol,
                            "price": current_price,
                            "option_volume": total_volume,
                            "open_interest": total_oi,
                            "put_call_ratio": put_call_ratio,
                            "nearest_expiry": exp_dates[0],
                            "next_earnings": earnings_info.get("next_earnings"),
                            "days_to_earnings": earnings_info.get("days_to_earnings"),
                        }
                        if avg_iv is not None:
                            entry.update(
                                {
                                    "atm_strike": atm_strike,
                                    "implied_volatility": avg_iv / 100,
                                    "implied_volatility_pct": avg_iv,
                                    "call_iv": atm_call_iv,
                                    "put_iv": atm_put_iv,
                                    "atm_metrics": {
                                        "strike": atm_strike,
                                        "call_spread": atm_call_spread,
                                        "put_spread": atm_put_spread,
                                        "avg_spread": avg_spread,
                                    },
                                }
                            )
                        results.append(entry)

                except Exception as e:
                    logger.error(f"Error processing options for {symbol}: {e}")
                    rejected.append({"symbol": symbol, "rejection_reasons": [f"Options error: {e}"]})

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                rejected.append({"symbol": symbol, "error": str(e)})

        return {
            "screen_type": "options",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Custom screen
    # ------------------------------------------------------------------

    async def _run_custom(self, symbols: list[str], criteria: dict) -> dict:
        results = []
        rejected = []

        technical_criteria = criteria.get("technical", {})
        fundamental_criteria = criteria.get("fundamental", {})
        options_criteria = criteria.get("options", {})
        news_criteria = criteria.get("news", {})

        for symbol in symbols:
            try:
                rejection_reasons: list[str] = []
                symbol_data: dict = {"symbol": symbol}

                if technical_criteria:
                    tech = await self._screen_single_technical(symbol, technical_criteria)
                    rejection_reasons.extend(tech.get("rejection_reasons", []))
                    symbol_data.update(tech.get("data", {}))

                if fundamental_criteria and not rejection_reasons:
                    fund = await self._screen_single_fundamental(symbol, fundamental_criteria)
                    rejection_reasons.extend(fund.get("rejection_reasons", []))
                    symbol_data.update(fund.get("data", {}))

                if options_criteria and not rejection_reasons:
                    opt = await self._screen_single_options(symbol, options_criteria)
                    rejection_reasons.extend(opt.get("rejection_reasons", []))
                    symbol_data.update(opt.get("data", {}))

                if news_criteria and not rejection_reasons:
                    news_result = await self.news_service.get_news_data(
                        symbol, days_back=news_criteria.get("max_days", 30)
                    )
                    if "error" in news_result:
                        rejection_reasons.append(f"News error: {news_result['error']}")
                    else:
                        all_news = (
                            news_result.get("recent_news", [])
                            + news_result.get("key_events", [])
                            + news_result.get("management_changes", [])
                        )
                        keywords = news_criteria.get("keywords", [])
                        matching_news = [
                            n for n in all_news
                            if not keywords
                            or any(
                                kw.lower() in f"{n['title']} {n.get('summary', '')}".lower()
                                for kw in keywords
                            )
                        ]
                        if news_criteria.get("management_changes") and not news_result.get(
                            "management_changes"
                        ):
                            rejection_reasons.append("No recent management changes found")
                        elif matching_news:
                            symbol_data["news"] = matching_news
                            symbol_data["management"] = news_result.get("current_management")
                        else:
                            rejection_reasons.append("No matching news found")

                if rejection_reasons:
                    symbol_data["rejection_reasons"] = rejection_reasons
                    rejected.append(symbol_data)
                else:
                    results.append(symbol_data)

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                rejected.append({"symbol": symbol, "error": str(e)})

        return {
            "screen_type": "custom",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Single-symbol helpers (used by custom screen)
    # ------------------------------------------------------------------

    async def _screen_single_technical(self, symbol: str, criteria: dict) -> dict:
        rejection_reasons: list[str] = []
        data: dict = {}
        try:
            info = await self.provider.get_ticker_info(symbol)
            hist = await self.provider.get_history(symbol)

            if hist.empty:
                return {"data": {}, "rejection_reasons": ["No historical data available"]}

            current_price = (
                (info or {}).get("regularMarketPrice")
                or (info or {}).get("currentPrice")
                or hist["Close"].iloc[-1]
            )
            if pd.isna(current_price):
                return {"data": {}, "rejection_reasons": ["No valid price data"]}

            data["price"] = current_price

            if "Volume" in hist.columns:
                avg_vol = hist["Volume"].mean()
                if not pd.isna(avg_vol):
                    data["volume"] = avg_vol
                    if "min_volume" in criteria and avg_vol < criteria["min_volume"]:
                        rejection_reasons.append(
                            f"Volume ({avg_vol:.0f}) < minimum ({criteria['min_volume']})"
                        )

            if "min_price" in criteria and current_price < criteria["min_price"]:
                rejection_reasons.append(
                    f"Price ({current_price:.2f}) < minimum ({criteria['min_price']})"
                )
            if "max_price" in criteria and current_price > criteria["max_price"]:
                rejection_reasons.append(
                    f"Price ({current_price:.2f}) > maximum ({criteria['max_price']})"
                )

            if criteria.get("above_sma_200") or criteria.get("above_sma_50"):
                sma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
                sma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
                data["sma_50"] = None if pd.isna(sma_50) else float(sma_50)
                data["sma_200"] = None if pd.isna(sma_200) else float(sma_200)
                if criteria.get("above_sma_200") and not pd.isna(sma_200) and current_price <= sma_200:
                    rejection_reasons.append("Price below SMA200")
                if criteria.get("above_sma_50") and not pd.isna(sma_50) and current_price <= sma_50:
                    rejection_reasons.append("Price below SMA50")

        except Exception as e:
            return {"data": {}, "rejection_reasons": [f"Technical analysis error: {e}"]}

        return {"data": data, "rejection_reasons": rejection_reasons}

    async def _screen_single_fundamental(self, symbol: str, criteria: dict) -> dict:
        rejection_reasons: list[str] = []
        data: dict = {}
        try:
            info = await self.provider.get_ticker_info(symbol)
            if not info:
                return {"data": {}, "rejection_reasons": ["No fundamental data available"]}

            market_cap = info.get("marketCap", 0)
            data["market_cap"] = market_cap
            if "min_market_cap" in criteria and market_cap < criteria["min_market_cap"]:
                rejection_reasons.append(
                    f"Market cap ({market_cap}) < minimum ({criteria['min_market_cap']})"
                )

            pe_ratio = info.get("forwardPE", 0)
            data["pe_ratio"] = pe_ratio
            if "min_pe" in criteria and pe_ratio < criteria["min_pe"]:
                rejection_reasons.append(f"P/E ({pe_ratio}) < minimum ({criteria['min_pe']})")
            if "max_pe" in criteria and pe_ratio > criteria["max_pe"]:
                rejection_reasons.append(f"P/E ({pe_ratio}) > maximum ({criteria['max_pe']})")

        except Exception as e:
            return {"data": {}, "rejection_reasons": [f"Fundamental analysis error: {e}"]}

        return {"data": data, "rejection_reasons": rejection_reasons}

    async def _screen_single_options(self, symbol: str, criteria: dict) -> dict:
        rejection_reasons: list[str] = []
        data: dict = {}
        try:
            exp_dates = await self.provider.get_option_expirations(symbol)
            if not exp_dates:
                return {"data": {}, "rejection_reasons": ["No options data available"]}

            chain = await self.provider.get_option_chain(symbol, exp_dates[0])
            if not chain or not hasattr(chain, "calls") or not hasattr(chain, "puts"):
                return {"data": {}, "rejection_reasons": ["Invalid options chain data"]}

            total_volume = int(chain.calls["volume"].sum() + chain.puts["volume"].sum())
            data["option_volume"] = total_volume
            if "min_option_volume" in criteria and total_volume < criteria["min_option_volume"]:
                rejection_reasons.append(
                    f"Option volume ({total_volume}) < minimum ({criteria['min_option_volume']})"
                )

            if "min_days_to_earnings" in criteria or "max_days_to_earnings" in criteria:
                earnings_info = await self.provider.get_earnings_dates(symbol)
                days = earnings_info.get("days_to_earnings")
                data["days_to_earnings"] = days
                if "min_days_to_earnings" in criteria and (
                    days is None or days < criteria["min_days_to_earnings"]
                ):
                    rejection_reasons.append(f"Days to earnings ({days}) < minimum")
                if "max_days_to_earnings" in criteria and (
                    days is None or days > criteria["max_days_to_earnings"]
                ):
                    rejection_reasons.append(f"Days to earnings ({days}) > maximum")

        except Exception as e:
            return {"data": {}, "rejection_reasons": [f"Options analysis error: {e}"]}

        return {"data": data, "rejection_reasons": rejection_reasons}
