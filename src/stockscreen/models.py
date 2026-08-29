from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScreenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    screen_type: Literal["technical", "fundamental", "options", "news", "custom"]
    criteria: dict[str, Any]
    watchlist: str | None = None
    save_result: str | None = None


class NewsInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9.^=-]+$")
    days_back: int = Field(default=30, ge=1, le=365)

    @field_validator("symbol")
    @classmethod
    def norm(cls, v):
        return v.strip().upper()


class WatchlistInput(BaseModel):
    action: Literal["create", "update", "delete", "get"]
    name: str
    symbols: list[str] | None = None
