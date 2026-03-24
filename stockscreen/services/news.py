"""News service — fetch, categorize, and screen news."""

import datetime
import logging

from stockscreen.providers.facade import MarketDataFacade

logger = logging.getLogger("stockscreen-server-v1")

_MANAGEMENT_KEYWORDS = ("ceo", "chief", "executive", "president", "chairman")
_KEY_EVENT_KEYWORDS = ("lawsuit", "investigation", "sec", "probe")


class NewsService:
    """Service for fetching and screening stock news."""

    def __init__(self, provider: MarketDataFacade):
        self.provider = provider

    async def get_news_data(self, symbol: str, days_back: int = 30) -> dict:
        """Get recent news data for a symbol, categorized by type."""
        try:
            news_data = {
                "recent_news": [],
                "key_events": [],
                "management_changes": [],
                "last_updated": datetime.datetime.now().isoformat(),
            }

            # Fetch news
            try:
                news = await self.provider.get_news(symbol)
                if news:
                    cutoff = datetime.datetime.now() - datetime.timedelta(days=days_back)
                    for item in news:
                        pub_time = datetime.datetime.fromtimestamp(
                            item["providerPublishTime"]
                        )
                        if pub_time < cutoff:
                            continue

                        news_item = {
                            "title": item.get("title"),
                            "publisher": item.get("publisher"),
                            "published_at": pub_time.isoformat(),
                            "type": item.get("type"),
                            "summary": item.get("summary"),
                        }

                        title_lower = news_item["title"].lower()
                        if any(t in title_lower for t in _MANAGEMENT_KEYWORDS):
                            news_data["management_changes"].append(news_item)
                        elif any(t in title_lower for t in _KEY_EVENT_KEYWORDS):
                            news_data["key_events"].append(news_item)
                        else:
                            news_data["recent_news"].append(news_item)
            except Exception as e:
                logger.warning(f"Error fetching news for {symbol}: {e}")

            # Enrich with company info
            try:
                info = await self.provider.get_ticker_info(symbol)
                if info:
                    if "companyOfficers" in info:
                        news_data["current_management"] = [
                            {
                                "name": officer.get("name"),
                                "title": officer.get("title"),
                                "since": officer.get("yearStarted"),
                            }
                            for officer in info["companyOfficers"]
                        ]

                    news_data["company_info"] = {
                        "description": info.get("longBusinessSummary"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "website": info.get("website"),
                        "last_updated": datetime.datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.warning(f"Error fetching info for {symbol}: {e}")

            return news_data

        except Exception as e:
            logger.error(f"Error in get_news_data for {symbol}: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.datetime.now().isoformat(),
            }

    async def screen_news(self, symbols: list[str], criteria: dict) -> dict:
        """Screen stocks based on news criteria.

        Criteria keys:
            keywords: list of keywords to match
            exclude_keywords: list of keywords to exclude
            require_all_keywords: bool — require all keywords (AND vs OR)
            min_days: int — minimum age of news
            max_days: int — maximum age of news
            management_changes: bool — require management changes
        """
        results = []
        rejected = []

        for symbol in symbols:
            try:
                news_data = await self.get_news_data(
                    symbol, days_back=criteria.get("max_days", 30)
                )

                if "error" in news_data:
                    rejected.append({"symbol": symbol, "error": news_data["error"]})
                    continue

                # Combine all news for filtering
                all_news = (
                    news_data.get("recent_news", [])
                    + news_data.get("key_events", [])
                    + news_data.get("management_changes", [])
                )

                # Filter by date range
                min_date = datetime.datetime.now() - datetime.timedelta(
                    days=criteria.get("max_days", 30)
                )
                max_date = datetime.datetime.now() - datetime.timedelta(
                    days=criteria.get("min_days", 0)
                )

                filtered_news = [
                    n
                    for n in all_news
                    if min_date
                    <= datetime.datetime.fromisoformat(n["published_at"])
                    <= max_date
                ]

                # Apply keyword filters
                keywords = criteria.get("keywords", [])
                exclude_keywords = criteria.get("exclude_keywords", [])
                require_all = criteria.get("require_all_keywords", False)

                matching_news = []
                for news in filtered_news:
                    text = f"{news['title']} {news['summary']}".lower()

                    if any(kw.lower() in text for kw in exclude_keywords):
                        continue

                    if keywords:
                        if require_all:
                            if all(kw.lower() in text for kw in keywords):
                                matching_news.append(news)
                        else:
                            if any(kw.lower() in text for kw in keywords):
                                matching_news.append(news)
                    else:
                        matching_news.append(news)

                # Check management changes requirement
                if criteria.get("management_changes"):
                    if not news_data.get("management_changes"):
                        rejected.append(
                            {"symbol": symbol, "reason": "No recent management changes found"}
                        )
                        continue

                if matching_news:
                    results.append(
                        {
                            "symbol": symbol,
                            "matching_news": matching_news,
                            "management": news_data.get("current_management"),
                            "company_info": news_data.get("company_info"),
                        }
                    )
                else:
                    rejected.append(
                        {"symbol": symbol, "reason": "No matching news found"}
                    )

            except Exception as e:
                logger.error(f"Error screening news for {symbol}: {e}")
                rejected.append({"symbol": symbol, "error": str(e)})

        return {
            "screen_type": "news",
            "criteria": criteria,
            "matches": len(results),
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.datetime.now().isoformat(),
        }
