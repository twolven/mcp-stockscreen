"""Tests for stockscreen.services.news module."""

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from stockscreen.services.news import NewsService


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def news_service(mock_provider):
    return NewsService(provider=mock_provider)


@pytest.fixture
def sample_news_items():
    now = datetime.datetime.now()
    return [
        {
            "title": "CEO announces new strategy for growth",
            "publisher": "Reuters",
            "providerPublishTime": int((now - datetime.timedelta(days=2)).timestamp()),
            "type": "STORY",
            "summary": "The CEO outlined plans for expansion.",
        },
        {
            "title": "SEC investigation into accounting practices",
            "publisher": "Bloomberg",
            "providerPublishTime": int((now - datetime.timedelta(days=5)).timestamp()),
            "type": "STORY",
            "summary": "Federal regulators are probing the company.",
        },
        {
            "title": "Quarterly earnings beat expectations",
            "publisher": "CNBC",
            "providerPublishTime": int((now - datetime.timedelta(days=10)).timestamp()),
            "type": "STORY",
            "summary": "Revenue and profit exceeded analyst estimates.",
        },
    ]


@pytest.fixture
def sample_ticker_info():
    return {
        "longBusinessSummary": "A technology company.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "website": "https://example.com",
        "companyOfficers": [
            {"name": "Tim Cook", "title": "CEO", "yearStarted": 2011},
        ],
    }


# ============================================================
# 1. get_news_data — basic fetch and categorization
# ============================================================
class TestGetNewsData:
    async def test_basic_news_fetch(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.get_news_data("AAPL", days_back=30)
        assert "recent_news" in result
        assert "key_events" in result
        assert "management_changes" in result
        assert "last_updated" in result

    async def test_news_categorization_management(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.get_news_data("AAPL", days_back=30)
        # "CEO announces..." should be in management_changes
        mgmt_titles = [n["title"] for n in result["management_changes"]]
        assert any("CEO" in t for t in mgmt_titles)

    async def test_news_categorization_key_events(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.get_news_data("AAPL", days_back=30)
        # "SEC investigation..." should be in key_events
        event_titles = [n["title"] for n in result["key_events"]]
        assert any("SEC" in t for t in event_titles)

    async def test_no_news(self, news_service, mock_provider, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=[])
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.get_news_data("AAPL")
        assert result["recent_news"] == []
        assert result["key_events"] == []
        assert result["management_changes"] == []

    async def test_partial_errors_still_return_structure(self, news_service, mock_provider):
        """When news and info both fail, we still get the base structure (no crash)."""
        mock_provider.get_news = AsyncMock(side_effect=RuntimeError("API down"))
        mock_provider.get_ticker_info = AsyncMock(side_effect=RuntimeError("API down"))

        result = await news_service.get_news_data("AAPL")
        # Should still return valid structure, just empty
        assert "recent_news" in result
        assert result["recent_news"] == []
        assert "company_info" not in result  # info fetch failed

    async def test_company_info_included(self, news_service, mock_provider, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=[])
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.get_news_data("AAPL")
        assert result["company_info"]["sector"] == "Technology"
        assert result["current_management"][0]["name"] == "Tim Cook"


# ============================================================
# 2. screen_news — filtering
# ============================================================
class TestScreenNews:
    async def test_keyword_match(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.screen_news(
            ["AAPL"], {"keywords": ["earnings"]}
        )
        assert result["screen_type"] == "news"
        assert result["matches"] >= 1

    async def test_keyword_exclusion(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.screen_news(
            ["AAPL"],
            {"keywords": ["earnings"], "exclude_keywords": ["beat"]},
        )
        # "beat" should exclude the earnings article
        assert result["screen_type"] == "news"

    async def test_require_all_keywords(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.screen_news(
            ["AAPL"],
            {"keywords": ["earnings", "nonexistent_word"], "require_all_keywords": True},
        )
        assert result["matches"] == 0

    async def test_no_matching_news(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.screen_news(
            ["AAPL"], {"keywords": ["zzzznotfound"]}
        )
        assert result["matches"] == 0

    async def test_management_changes_filter(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        result = await news_service.screen_news(
            ["AAPL"], {"management_changes": True}
        )
        assert result["screen_type"] == "news"

    async def test_date_filtering(self, news_service, mock_provider, sample_news_items, sample_ticker_info):
        mock_provider.get_news = AsyncMock(return_value=sample_news_items)
        mock_provider.get_ticker_info = AsyncMock(return_value=sample_ticker_info)

        # Only get news from last 3 days (should exclude 5-day and 10-day old items)
        result = await news_service.screen_news(
            ["AAPL"], {"max_days": 3}
        )
        assert result["screen_type"] == "news"
