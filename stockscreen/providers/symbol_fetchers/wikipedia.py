"""Wikipedia-based symbol fetchers.

Each fetcher reads an index table from Wikipedia via pandas.read_html
(executed in a thread pool to stay non-blocking) and returns a list of
SymbolRecord instances.
"""

import asyncio
import logging
from typing import ClassVar

import pandas as pd

from stockscreen.exceptions import APIError
from stockscreen.providers.symbol_fetchers.base import BaseSymbolFetcher, SymbolRecord

logger = logging.getLogger("stockscreen-server-v1")


class _WikipediaFetcher(BaseSymbolFetcher):
    """Shared fetch logic for all Wikipedia index pages.

    Subclasses must set:
        name, source_url          — required by BaseSymbolFetcher
        _symbol_col (str)         — column name for the ticker
        _name_col   (str)         — column name for the company name
        _table_index (int)        — which table on the page (default 0)
        _suffix     (str)         — exchange suffix to append, e.g. ".PA"
    """

    _symbol_col: ClassVar[str]
    _name_col: ClassVar[str]
    _table_index: ClassVar[int] = 0
    _suffix: ClassVar[str] = ""

    async def fetch(self) -> list[SymbolRecord]:
        loop = asyncio.get_event_loop()
        try:
            tables: list[pd.DataFrame] = await loop.run_in_executor(
                None, lambda: pd.read_html(self.source_url)
            )
        except Exception as exc:
            raise APIError(f"[{self.name}] Failed to fetch {self.source_url}: {exc}") from exc

        try:
            df = tables[self._table_index]
            return self._parse(df)
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"[{self.name}] Failed to parse table: {exc}") from exc

    def _parse(self, df: pd.DataFrame) -> list[SymbolRecord]:
        records: list[SymbolRecord] = []
        for _, row in df.iterrows():
            raw = str(row.get(self._symbol_col, "")).strip()
            if not raw or raw.lower() == "nan":
                continue
            symbol = self._apply_suffix(raw)
            name = str(row.get(self._name_col, "")).strip()
            records.append(SymbolRecord(symbol=symbol, name=name, instrument_type="equity"))
        return records

    def _apply_suffix(self, symbol: str) -> str:
        if self._suffix and not symbol.endswith(self._suffix):
            return symbol + self._suffix
        return symbol


# ---------------------------------------------------------------------------
# Concrete fetchers
# ---------------------------------------------------------------------------

class SP500Fetcher(_WikipediaFetcher):
    name = "sp500"
    source_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    _symbol_col = "Symbol"
    _name_col = "Security"
    _suffix = ""


class Nasdaq100Fetcher(_WikipediaFetcher):
    name = "nasdaq100"
    source_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    _symbol_col = "Ticker"
    _name_col = "Company"
    _suffix = ""


class CAC40Fetcher(_WikipediaFetcher):
    name = "cac40"
    source_url = "https://en.wikipedia.org/wiki/CAC_40"
    _symbol_col = "Ticker symbol"
    _name_col = "Company"
    _suffix = ".PA"


class SBF120Fetcher(_WikipediaFetcher):
    name = "sbf120"
    source_url = "https://en.wikipedia.org/wiki/SBF_120"
    _symbol_col = "Ticker symbol"
    _name_col = "Company"
    _suffix = ".PA"


class DAXFetcher(_WikipediaFetcher):
    name = "dax"
    source_url = "https://en.wikipedia.org/wiki/DAX"
    _symbol_col = "Ticker symbol"
    _name_col = "Company"
    _suffix = ".DE"


class FTSE100Fetcher(_WikipediaFetcher):
    name = "ftse100"
    source_url = "https://en.wikipedia.org/wiki/FTSE_100"
    _symbol_col = "Ticker"
    _name_col = "Company"
    _suffix = ".L"


class AEXFetcher(_WikipediaFetcher):
    name = "aex"
    source_url = "https://en.wikipedia.org/wiki/AEX_index"
    _symbol_col = "Ticker symbol"
    _name_col = "Company"
    _suffix = ".AS"
