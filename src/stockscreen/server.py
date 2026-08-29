from datetime import UTC, date, datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .domain import days_until_earnings, fundamental, news_matches, options_metrics, technical
from .models import (
    Action,
    Criteria,
    DaysBack,
    Name,
    ResponseEnvelope,
    ScreenType,
    Symbol,
    validate_criteria,
)
from .persistence import Store
from .provider import ProviderError, YahooProvider, clean, envelope, normalize_news

mcp = FastMCP("stockscreen")
provider = YahooProvider()
store = Store()


def symbols_from(criteria, watchlist):
    if watchlist:
        symbols = store.load("watchlist", watchlist)
        if symbols is None:
            raise ToolError(f"Watchlist '{watchlist}' not found")
        return symbols, {
            "source": "watchlist",
            "name": watchlist,
            "as_of": datetime.now(UTC).isoformat(),
        }
    raw = criteria.get("symbols")
    if raw:
        return ([raw] if isinstance(raw, str) else raw), {
            "source": "explicit",
            "as_of": datetime.now(UTC).isoformat(),
        }
    symbols = provider.universe(250, criteria.get("category"))
    return symbols, {
        "source": "Yahoo EquityQuery",
        "region": "us",
        "sort": "assets/market-cap desc",
        "category": criteria.get("category"),
        "requested_size": 250,
        "returned_size": len(symbols),
        "as_of": datetime.now(UTC).isoformat(),
    }


def evaluate(symbol, screen_type, criteria):
    ticker = provider.ticker(symbol)
    info = provider.info(symbol, ticker) or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return False, ["invalid or missing quote"], {}
    if screen_type == "fundamental":
        return fundamental(info, criteria)
    if screen_type == "technical":
        return technical(provider.history(symbol, ticker), criteria)
    if screen_type == "options":
        expirations = provider.expirations(symbol, ticker)
        if not expirations:
            return False, ["no option expirations"], {}
        chain = provider.chain(symbol, expirations[0], ticker)
        days = days_until_earnings(provider.calendar(symbol, ticker))
        expiration_days = (date.fromisoformat(expirations[0]) - datetime.now(UTC).date()).days
        return options_metrics(chain.calls, chain.puts, criteria, days, expiration_days)
    if screen_type == "news":
        articles = normalize_news(provider.news(symbol, ticker), int(criteria.get("max_days", 30)))
        return news_matches(articles, criteria)
    sections = {}
    reasons = []
    for category in ("technical", "fundamental", "options", "news"):
        nested = criteria.get(category, {})
        if nested:
            ok, section_reasons, details = evaluate(symbol, category, nested)
            sections[category] = details
            if not ok:
                reasons.extend(f"{category}: {reason}" for reason in section_reasons)
    return not reasons, reasons, sections


@mcp.tool
def run_stock_screen(
    screen_type: ScreenType,
    criteria: Criteria,
    watchlist: Name | None = None,
    save_result: Name | None = None,
) -> ResponseEnvelope:
    """Run a legacy technical, fundamental, options, news, or custom screen."""
    try:
        validate_criteria(screen_type, criteria)
        symbols, universe = symbols_from(criteria, watchlist)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        try:
            ok, reasons, details = evaluate(symbol, screen_type, criteria)
            (accepted if ok else rejected).append(
                {"symbol": symbol, "details": details}
                if ok
                else {"symbol": symbol, "reasons": reasons}
            )
        except (ProviderError, TimeoutError, ConnectionError, OSError, ValueError, KeyError) as exc:
            rejected.append({"symbol": symbol, "reasons": [f"provider error: {exc}"]})
    result = {
        "screen_type": screen_type,
        "criteria": criteria,
        "universe": universe,
        "universe_size": len(symbols),
        "matches": accepted,
        "rejected": rejected,
    }
    if save_result:
        try:
            store.save("result", save_result, clean(result))
        except (ValueError, TypeError) as exc:
            raise ToolError(str(exc)) from exc
    return envelope(
        result,
        [
            "Yahoo Finance data may be delayed, incomplete, or rate-limited; this is not investment advice."
        ],
    )


@mcp.tool
def get_stock_news(symbol: Symbol, days_back: DaysBack = 30) -> ResponseEnvelope:
    """Get normalized recent Yahoo Finance news."""
    normalized = symbol.strip().upper()
    ticker = provider.ticker(normalized)
    try:
        items = normalize_news(provider.news(normalized, ticker), days_back)
    except (ProviderError, TimeoutError, ConnectionError, OSError) as exc:
        raise ToolError(f"Yahoo Finance request failed after bounded retries: {exc}") from exc
    return envelope({"symbol": normalized, "days_back": days_back, "articles": items})


@mcp.tool
def manage_watchlist(
    action: Action, name: Name, symbols: list[Symbol] | None = None
) -> ResponseEnvelope:
    """Create, update, delete, or retrieve a safely persisted watchlist."""
    if action in ("create", "update"):
        if not symbols:
            raise ToolError("symbols required for create/update")
        normalized = list(dict.fromkeys(value.strip().upper() for value in symbols))
        store.save("watchlist", name, normalized)
        data = {"name": name, "symbols": normalized}
    elif action == "delete":
        if not store.delete("watchlist", name):
            raise ToolError(f"Watchlist '{name}' not found")
        data = {"message": f"Watchlist '{name}' deleted"}
    else:
        try:
            value = store.load("watchlist", name)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        if value is None:
            raise ToolError(f"Watchlist '{name}' not found")
        data = {"name": name, "symbols": value}
    return envelope(data)


@mcp.tool
def get_screening_result(name: Name) -> ResponseEnvelope:
    """Retrieve a saved screening result."""
    try:
        value = store.load("result", name)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if value is None:
        raise ToolError(f"Screening result '{name}' not found")
    return envelope(value)


def main():
    mcp.run(transport="stdio", show_banner=False)
