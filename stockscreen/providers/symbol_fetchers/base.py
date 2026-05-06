"""Base interface for symbol fetchers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


_VALID_INSTRUMENT_TYPES = {"equity", "etf"}


@dataclass
class SymbolRecord:
    """A single symbol returned by a fetcher.

    Attributes:
        symbol: Ticker symbol as used by Yahoo Finance (e.g. "TTE.PA").
        name: Human-readable company or fund name.
        market_cap: Market capitalisation in USD, or None if unknown.
        instrument_type: "equity" | "etf" | None.
    """

    symbol: str
    name: str
    market_cap: float | None = field(default=None)
    instrument_type: str | None = field(default=None)

    def __post_init__(self):
        self.symbol = self.symbol.strip().upper()
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.market_cap is not None and self.market_cap < 0:
            raise ValueError("market_cap must be a positive number")
        if self.instrument_type is not None and self.instrument_type not in _VALID_INSTRUMENT_TYPES:
            raise ValueError(
                f"instrument_type must be one of {_VALID_INSTRUMENT_TYPES}, "
                f"got '{self.instrument_type}'"
            )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market_cap": self.market_cap,
            "instrument_type": self.instrument_type,
        }


class BaseSymbolFetcher(ABC):
    """Abstract base class for all symbol fetchers.

    Subclasses must define two string class attributes:
        name (str):       Short identifier used as cache key (e.g. "cac40").
        source_url (str): URL of the data source (for logging and docs).

    Subclasses must implement:
        fetch() -> list[SymbolRecord]
    """

    name: ClassVar[str]
    source_url: ClassVar[str]

    def __init__(self):
        # Enforce class-level string attributes on concrete subclasses.
        # (Abstract classes are already blocked from instantiation by ABC.)
        for attr in ("name", "source_url"):
            if not isinstance(getattr(type(self), attr, None), str):
                raise TypeError(
                    f"{type(self).__name__} must define a string class attribute '{attr}'"
                )

    @abstractmethod
    async def fetch(self) -> list[SymbolRecord]:
        """Fetch the symbol list from the source.

        Returns:
            A list of SymbolRecord instances. May be empty if the source
            returns no data, but must not raise on an empty result.

        Raises:
            APIError: On unrecoverable network or parse failures.
        """
