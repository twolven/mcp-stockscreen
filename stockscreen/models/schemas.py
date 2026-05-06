"""Pydantic models and JSON encoder for stockscreen."""

import datetime
import json
import re

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class WatchlistName(BaseModel):
    """Validated watchlist name."""

    name: str = Field(min_length=1, max_length=50)

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$", v):
            raise ValueError(
                "Watchlist name can only contain letters, numbers, underscore, "
                "and hyphen, and cannot start with a hyphen"
            )
        return v


class StockSymbols(BaseModel):
    """Validated list of stock symbols."""

    symbols: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        validated = []
        for symbol in v:
            if not isinstance(symbol, str):
                raise ValueError(f"All symbols must be strings, got {type(symbol)}")
            if not 1 <= len(symbol) <= 10:
                raise ValueError(f"Invalid symbol length: {symbol}")
            if not symbol.replace("-", "").replace(".", "").isalnum():
                raise ValueError(f"Invalid symbol format: {symbol}")
            validated.append(symbol.upper())
        return validated


class StockscreenJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles pandas and numpy types."""

    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj):
        """Replace float NaN/Inf with None before standard encoding."""
        if isinstance(obj, float) and (pd.isna(obj) or obj != obj):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj
