import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

MISSING = object()


def metric(info: dict[str, Any], key: str):
    value = info.get(key, MISSING)
    if value is MISSING or value is None:
        return MISSING
    try:
        value = float(value)
    except (TypeError, ValueError):
        return MISSING
    return value if math.isfinite(value) else MISSING


def _bound(value: float | object, criteria: dict[str, Any], name: str, reasons: list[str]) -> None:
    requested = f"min_{name}" in criteria or f"max_{name}" in criteria
    if not requested:
        return
    if value is MISSING:
        reasons.append(f"missing metric: {name}")
        return
    assert isinstance(value, float)
    if f"min_{name}" in criteria and value < float(criteria[f"min_{name}"]):
        reasons.append(f"failed min_{name}")
    if f"max_{name}" in criteria and value > float(criteria[f"max_{name}"]):
        reasons.append(f"failed max_{name}")


def fundamental(
    info: dict[str, Any], criteria: dict[str, Any]
) -> tuple[bool, list[str], dict[str, Any]]:
    mapping = {
        "market_cap": "marketCap",
        "pe": "trailingPE",
        "dividend": "dividendYield",
        "revenue_growth": "revenueGrowth",
        "aum": "totalAssets",
        "expense_ratio": "annualReportExpenseRatio",
        "volume": "regularMarketVolume",
    }
    reasons: list[str] = []
    values: dict[str, Any] = {}
    for name, key in mapping.items():
        value = metric(info, key)
        values[name] = None if value is MISSING else value
        _bound(value, criteria, name, reasons)
    return not reasons, reasons, values


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    output = pd.Series(float("nan"), index=close.index, dtype=float)
    if len(close) <= period:
        return output
    avg_gain, avg_loss = gains.iloc[1 : period + 1].mean(), losses.iloc[1 : period + 1].mean()

    def score(gain, loss):
        if gain == loss == 0:
            return 50.0
        if loss == 0:
            return 100.0
        if gain == 0:
            return 0.0
        return 100 - 100 / (1 + gain / loss)

    output.iloc[period] = score(avg_gain, avg_loss)
    for index in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[index]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[index]) / period
        output.iloc[index] = score(avg_gain, avg_loss)
    return output


def technical(
    history: pd.DataFrame, criteria: dict[str, Any]
) -> tuple[bool, list[str], dict[str, Any]]:
    if history.empty or "Close" not in history:
        return False, ["missing price history"], {}
    close = history.Close.astype(float)
    latest = float(close.iloc[-1])
    reasons: list[str] = []
    volume = (
        float(history.Volume.tail(20).mean())
        if "Volume" in history and history.Volume.notna().any()
        else None
    )
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi = _wilder_rsi(close).iloc[-1]
    rsi_value = None if pd.isna(rsi) else float(rsi)
    atr_pct = None
    if {"High", "Low"} <= set(history):
        high, low, previous = history.High.astype(float), history.Low.astype(float), close.shift(1)
        true_range = pd.concat(
            [(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        atr_pct = None if pd.isna(atr) else float(atr / latest * 100)
    for key, value, comparison in (
        ("min_price", latest, lambda a, b: a < b),
        ("max_price", latest, lambda a, b: a > b),
        ("min_volume", volume, lambda a, b: a < b),
        ("min_rsi", rsi_value, lambda a, b: a < b),
        ("max_rsi", rsi_value, lambda a, b: a > b),
        ("max_atr_pct", atr_pct, lambda a, b: a > b),
    ):
        if key in criteria:
            if value is None:
                reasons.append(f"missing metric: {key.removeprefix('min_').removeprefix('max_')}")
            elif comparison(value, float(criteria[key])):
                reasons.append(f"failed {key}")
    if criteria.get("above_sma_50") and (sma50 is None or latest <= sma50):
        reasons.append("not above SMA 50")
    if criteria.get("above_sma_200") and (sma200 is None or latest <= sma200):
        reasons.append("not above SMA 200")
    return (
        not reasons,
        reasons,
        {
            "price": latest,
            "average_volume": volume,
            "sma_50": sma50,
            "sma_200": sma200,
            "rsi": rsi_value,
            "atr_percent": atr_pct,
        },
    )


def options_metrics(
    calls: pd.DataFrame, puts: pd.DataFrame, criteria: dict[str, Any], days_to_earnings: int | None
) -> tuple[bool, list[str], dict[str, Any]]:
    combined = pd.concat([calls, puts], ignore_index=True)
    reasons: list[str] = []
    iv = (
        combined.impliedVolatility.dropna()
        if "impliedVolatility" in combined
        else pd.Series(dtype=float)
    )
    avg_iv = None if iv.empty else float(iv.mean() * 100)
    call_volume = float(calls.volume.fillna(0).sum()) if "volume" in calls else 0
    put_volume = float(puts.volume.fillna(0).sum()) if "volume" in puts else 0
    total_volume = call_volume + put_volume
    midpoint = (
        (combined.bid.fillna(0) + combined.ask.fillna(0)) / 2
        if {"bid", "ask"} <= set(combined)
        else pd.Series(dtype=float)
    )
    spreads = (
        ((combined.ask - combined.bid) / midpoint.where(midpoint > 0) * 100)
        .replace([math.inf, -math.inf], pd.NA)
        .dropna()
        if not midpoint.empty
        else pd.Series(dtype=float)
    )
    avg_spread = None if spreads.empty else float(spreads.mean())
    ratio = None if call_volume == 0 else put_volume / call_volume
    values = {
        "iv": avg_iv,
        "option_volume": total_volume,
        "put_call_ratio": ratio,
        "spread": avg_spread,
        "days_to_earnings": days_to_earnings,
    }
    for key in ("iv", "option_volume", "put_call_ratio", "days_to_earnings"):
        raw_value = values[key]
        _bound(MISSING if raw_value is None else float(raw_value), criteria, key, reasons)
    if "max_spread" in criteria and (
        avg_spread is None or avg_spread > float(criteria["max_spread"])
    ):
        reasons.append("missing or failed max_spread")
    return not reasons, reasons, values


def news_matches(
    articles: list[dict[str, Any]], criteria: dict[str, Any]
) -> tuple[bool, list[str], dict[str, Any]]:
    keywords = [str(x).lower() for x in criteria.get("keywords", [])]
    excluded = [str(x).lower() for x in criteria.get("exclude_keywords", [])]
    require_all = bool(criteria.get("require_all_keywords"))
    matches = []
    management_terms = ("ceo", "cfo", "executive", "management", "resign", "appointed")
    now = datetime.now(UTC)
    for article in articles:
        if "min_days" in criteria:
            try:
                published = datetime.fromisoformat(
                    str(article.get("published_at", "")).replace("Z", "+00:00")
                )
                if (now - published).total_seconds() < float(criteria["min_days"]) * 86400:
                    continue
            except ValueError:
                continue
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
        if excluded and any(word in text for word in excluded):
            continue
        if keywords and not (
            all(word in text for word in keywords)
            if require_all
            else any(word in text for word in keywords)
        ):
            continue
        if criteria.get("management_changes") and not any(
            word in text for word in management_terms
        ):
            continue
        matches.append(article)
    return bool(matches), [] if matches else ["no news matched criteria"], {"articles": matches}


def days_until_earnings(calendar: Any, now: datetime | None = None) -> int | None:
    current = (now or datetime.now(UTC)).date()
    raw = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    try:
        return (pd.Timestamp(raw).date() - current).days if raw is not None else None
    except (TypeError, ValueError):
        return None
