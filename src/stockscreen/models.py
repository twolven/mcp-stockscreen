from typing import Annotated, Any, Literal, TypedDict

from pydantic import Field

ScreenType = Annotated[
    Literal["technical", "fundamental", "options", "news", "custom"],
    Field(description="Legacy stock-screen category"),
]
Action = Annotated[
    Literal["create", "update", "delete", "get"], Field(description="Watchlist action")
]
Symbol = Annotated[
    str,
    Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9.^=-]+$",
        description="Yahoo Finance ticker symbol",
    ),
]
DaysBack = Annotated[int, Field(ge=1, le=365, description="Maximum news age in days")]
Name = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
        description="Safe persistence name",
    ),
]
Criteria = Annotated[
    dict[str, Any], Field(description="Criteria for the selected legacy screen category")
]


class ProviderMetadata(TypedDict):
    name: str
    as_of: str
    real_time: bool


class ResponseEnvelope(TypedDict):
    success: bool
    timestamp: str
    data: dict[str, Any]
    provider: ProviderMetadata
    warnings: list[str]


COMMON = {"symbols", "category"}
TECHNICAL = COMMON | {
    "min_price",
    "max_price",
    "min_volume",
    "above_sma_50",
    "above_sma_200",
    "min_rsi",
    "max_rsi",
    "max_atr_pct",
}
FUNDAMENTAL = COMMON | {
    "min_market_cap",
    "min_pe",
    "max_pe",
    "min_dividend",
    "min_revenue_growth",
    "min_aum",
    "max_expense_ratio",
    "min_volume",
}
OPTIONS = COMMON | {
    "min_iv",
    "max_iv",
    "min_option_volume",
    "min_put_call_ratio",
    "max_spread",
    "min_days_to_earnings",
    "max_days_to_earnings",
}
NEWS = COMMON | {
    "min_days",
    "max_days",
    "keywords",
    "exclude_keywords",
    "require_all_keywords",
    "management_changes",
}
CUSTOM = COMMON | {"technical", "fundamental", "options", "news"}
ALLOWED = {
    "technical": TECHNICAL,
    "fundamental": FUNDAMENTAL,
    "options": OPTIONS,
    "news": NEWS,
    "custom": CUSTOM,
}


def validate_criteria(screen_type: str, criteria: dict[str, Any]) -> None:
    unknown = set(criteria) - ALLOWED[screen_type]
    if unknown:
        raise ValueError(f"Unsupported {screen_type} criteria: {', '.join(sorted(unknown))}")
    if "category" in criteria and criteria["category"] not in {
        "mega_cap",
        "large_cap",
        "mid_cap",
        "small_cap",
        "micro_cap",
        "etf",
    }:
        raise ValueError(f"Invalid category: {criteria['category']}")
    if screen_type == "custom":
        for category in ("technical", "fundamental", "options", "news"):
            nested = criteria.get(category, {})
            if not isinstance(nested, dict):
                raise TypeError(f"custom.{category} must be an object")
            nested_unknown = set(nested) - ALLOWED[category]
            if nested_unknown:
                raise ValueError(
                    f"Unsupported custom.{category} criteria: {', '.join(sorted(nested_unknown))}"
                )
