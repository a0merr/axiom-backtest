"""Market data handlers.

A data handler streams bars one timestamp at a time so the strategy and
portfolio only ever see data up to "now". This is the single most important
guard against look-ahead bias: components cannot peek at future bars because
those rows have not been yielded yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from .event import MarketEvent


class DataHandler(ABC):
    """Interface for anything that feeds bars into the engine."""

    symbols: list[str]

    @abstractmethod
    def update_bars(self) -> MarketEvent | None:
        """Advance one timestamp. Return a MarketEvent, or None when exhausted."""

    @abstractmethod
    def latest_bar(self, symbol: str) -> pd.Series | None:
        """Most recent bar visible for ``symbol`` (never a future bar)."""

    @abstractmethod
    def latest_price(self, symbol: str, field: str = "close") -> float | None:
        """Most recent value of ``field`` for ``symbol``."""


class HistoricCSVDataHandler(DataHandler):
    """Drive a backtest from in-memory OHLCV frames.

    Each frame must be indexed by a sorted ``DatetimeIndex`` and contain at
    least a ``close`` column. The handler unions all symbols' timestamps so a
    sparse symbol simply carries its last-known bar forward.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not frames:
            raise ValueError("frames must contain at least one symbol")
        self.symbols = list(frames.keys())
        self._frames = {s: self._validate(s, df) for s, df in frames.items()}

        index = None
        for df in self._frames.values():
            index = df.index if index is None else index.union(df.index)
        assert index is not None  # guaranteed: frames is non-empty
        self._timeline: list[datetime] = list(index)
        self._cursor = -1
        self._latest: dict[str, pd.Series] = {}

    @staticmethod
    def _validate(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError(f"{symbol}: frame must have a 'close' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: index must be a DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def update_bars(self) -> MarketEvent | None:
        self._cursor += 1
        if self._cursor >= len(self._timeline):
            return None
        ts = self._timeline[self._cursor]
        for symbol, df in self._frames.items():
            # asof avoids look-ahead: only rows at or before ``ts`` are visible.
            pos = df.index.get_indexer([ts], method="ffill")[0]
            if pos != -1:
                self._latest[symbol] = df.iloc[pos]
        return MarketEvent(timestamp=ts)

    def latest_bar(self, symbol: str) -> pd.Series | None:
        return self._latest.get(symbol)

    def latest_price(self, symbol: str, field: str = "close") -> float | None:
        bar = self._latest.get(symbol)
        if bar is None:
            return None
        return float(bar[field])
