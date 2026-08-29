import math

import pandas as pd

MISSING = object()


def metric(info: dict, key: str):
    value = info.get(key, MISSING)
    if value is MISSING or value is None:
        return MISSING
    try:
        value = float(value)
    except (TypeError, ValueError):
        return MISSING
    return value if math.isfinite(value) else MISSING


def check(value: float, criteria: dict, prefix: str = "") -> list[str]:
    failed = []
    for suffix, op in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b)):
        key = f"{prefix}_{suffix}" if prefix else suffix
        if key in criteria and not op(value, float(criteria[key])):
            failed.append(key)
    return failed


def fundamental(info: dict, criteria: dict) -> tuple[bool, list[str]]:
    mapping = {
        "market_cap": "marketCap",
        "pe_ratio": "trailingPE",
        "dividend_yield": "dividendYield",
        "revenue_growth": "revenueGrowth",
        "profit_margin": "profitMargins",
        "debt_to_equity": "debtToEquity",
        "price_to_book": "priceToBook",
    }
    reasons = []
    for name, key in mapping.items():
        wanted = any(
            k in criteria for k in (f"min_{name}", f"max_{name}", f"{name}_min", f"{name}_max")
        )
        if not wanted:
            continue
        value = metric(info, key)
        if value is MISSING:
            reasons.append(f"missing metric: {name}")
            continue
        for bound in (f"min_{name}", f"{name}_min"):
            if bound in criteria and value < float(criteria[bound]):
                reasons.append(f"failed {bound}")
        for bound in (f"max_{name}", f"{name}_max"):
            if bound in criteria and value > float(criteria[bound]):
                reasons.append(f"failed {bound}")
    return not reasons, reasons


def technical(history: pd.DataFrame, criteria: dict) -> tuple[bool, list[str], dict]:
    if history.empty or "Close" not in history:
        return False, ["missing price history"], {}
    close = history.Close.astype(float)
    latest = float(close.iloc[-1])
    metrics = {
        "price": latest,
        "sma_20": float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None,
        "sma_50": float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None,
        "average_volume": float(history.Volume.tail(20).mean()) if "Volume" in history else None,
    }
    reasons = []
    if "min_price" in criteria and latest < float(criteria["min_price"]):
        reasons.append("failed min_price")
    if "max_price" in criteria and latest > float(criteria["max_price"]):
        reasons.append("failed max_price")
    if criteria.get("above_sma_20") and (metrics["sma_20"] is None or latest <= metrics["sma_20"]):
        reasons.append("not above SMA 20")
    return not reasons, reasons, metrics
