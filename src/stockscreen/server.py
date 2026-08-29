from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .domain import fundamental, technical
from .models import NewsInput, ScreenInput, WatchlistInput
from .persistence import Store
from .provider import YahooProvider, envelope, normalize_news

mcp = FastMCP("stockscreen")
provider = YahooProvider()
store = Store()


def symbols_from(criteria, watchlist):
    if watchlist:
        symbols = store.load("watchlist", watchlist)
        if symbols is None:
            raise ToolError(f"Watchlist '{watchlist}' not found")
        return symbols, "watchlist"
    raw = criteria.get("symbols")
    if raw:
        return ([raw] if isinstance(raw, str) else raw), "explicit"
    return provider.universe(250), "Yahoo EquityQuery (maximum 250 results)"


@mcp.tool
def run_stock_screen(
    screen_type: str, criteria: dict, watchlist: str | None = None, save_result: str | None = None
) -> dict:
    """Run a technical, fundamental, options, news, or custom stock screen."""
    args = ScreenInput(
        screen_type=screen_type, criteria=criteria, watchlist=watchlist, save_result=save_result
    )
    try:
        symbols, universe = symbols_from(args.criteria, args.watchlist)
    except (ValueError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc
    accepted = []
    rejected = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        try:
            ticker = provider.ticker(symbol)
            info = ticker.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                rejected.append({"symbol": symbol, "reasons": ["invalid or missing quote"]})
                continue
            if args.screen_type == "fundamental":
                ok, reasons = fundamental(info, args.criteria)
                details = info
            elif args.screen_type == "technical":
                ok, reasons, details = technical(
                    ticker.history(period="1y", auto_adjust=False, repair=True, timeout=15),
                    args.criteria,
                )
            elif args.screen_type == "options":
                expirations = list(ticker.options or [])
                ok = bool(expirations)
                reasons = [] if ok else ["no option expirations"]
                details = {"expiration_dates": expirations}
            elif args.screen_type == "news":
                news = normalize_news(ticker.news, args.criteria.get("days_back", 30))
                ok = bool(news)
                reasons = [] if ok else ["no matching news"]
                details = {"news": news}
            else:
                okf, rf = fundamental(info, args.criteria)
                okt, rt, td = technical(
                    ticker.history(period="1y", auto_adjust=False, repair=True, timeout=15),
                    args.criteria,
                )
                ok = okf and okt
                reasons = rf + rt
                details = {"fundamental": info, "technical": td}
            (accepted if ok else rejected).append(
                {"symbol": symbol, "details": details}
                if ok
                else {"symbol": symbol, "reasons": reasons}
            )
        except Exception as exc:
            rejected.append({"symbol": symbol, "reasons": [f"provider error: {exc}"]})
    result = {
        "screen_type": args.screen_type,
        "criteria": args.criteria,
        "universe": universe,
        "universe_size": len(symbols),
        "matches": accepted,
        "rejected": rejected,
    }
    if args.save_result:
        try:
            store.save("result", args.save_result, result)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
    return envelope(
        result,
        [
            "Yahoo Finance data may be delayed, incomplete, or rate-limited; this is not investment advice."
        ],
    )


@mcp.tool
def get_stock_news(symbol: str, days_back: int = 30) -> dict:
    """Get normalized recent Yahoo Finance news."""
    args = NewsInput(symbol=symbol, days_back=days_back)
    try:
        items = normalize_news(provider.ticker(args.symbol).news, args.days_back)
    except Exception as exc:
        raise ToolError(f"Yahoo Finance request failed: {exc}") from exc
    return envelope({"symbol": args.symbol, "days_back": args.days_back, "articles": items})


@mcp.tool
def manage_watchlist(action: str, name: str, symbols: list[str] | None = None) -> dict:
    """Create, update, delete, or retrieve a safely persisted watchlist."""
    args = WatchlistInput(action=action, name=name, symbols=symbols)
    try:
        if args.action in ("create", "update"):
            if not args.symbols:
                raise ToolError("symbols required for create/update")
            normalized = list(dict.fromkeys(s.strip().upper() for s in args.symbols if s.strip()))
            store.save("watchlist", args.name, normalized)
            data = {"name": args.name, "symbols": normalized}
        elif args.action == "delete":
            if not store.delete("watchlist", args.name):
                raise ToolError(f"Watchlist '{args.name}' not found")
            data = {"message": f"Watchlist '{args.name}' deleted"}
        else:
            value = store.load("watchlist", args.name)
            if value is None:
                raise ToolError(f"Watchlist '{args.name}' not found")
            data = {"name": args.name, "symbols": value}
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return envelope(data)


@mcp.tool
def get_screening_result(name: str) -> dict:
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
