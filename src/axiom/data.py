"""Market data handlers.

A data handler streams bars one timestamp at a time so the strategy and
portfolio only ever see data up to "now". This is the single most important
guard against look-ahead bias: components cannot peek at future bars because
those rows have not been yielded yet.

Streaming honestly does not require streaming slowly. The naive way to write
this is to ask pandas for "the row at or before ``ts``" on every bar, for every
symbol. That is correct and roughly a hundred times slower than it needs to be:
each call re-validates the timestamp, re-runs a fill search, and materializes a
fresh ``Series``. All of that work is either loop-invariant or pure ceremony.

So alignment is computed **once**, vectorized, at construction: for every symbol
we precompute the row index it should show at every point on the timeline
(``-1`` before its first bar). The hot loop then advances an integer cursor and
reads from a contiguous ``float64`` matrix — no pandas calls, no allocations.
The visibility guarantee is unchanged, because the precomputed indices are
themselves backward-looking: position ``i`` on the timeline can only ever map to
a row at or before timeline time ``i``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

import numpy as np
import pandas as pd

from .event import MarketEvent


class Bar:
    """A read-only view of one row of OHLCV data.

    Behaves like the ``pd.Series`` it replaces for the operations the engine
    actually performs — ``bar["close"]``, ``"volume" in bar``, iteration over
    field names — but costs a pointer instead of a pandas object. Field access
    returns a plain ``float``, not a ``np.float64``, because the downstream
    arithmetic is scalar Python and the boxing shows up in profiles.

    Use :meth:`to_series` when you genuinely need pandas semantics.
    """

    __slots__ = ("_values", "_columns", "_row", "timestamp", "symbol")

    def __init__(
        self,
        values: np.ndarray,
        columns: dict[str, int],
        row: int,
        timestamp: datetime,
        symbol: str,
    ):
        self._values = values
        self._columns = columns
        self._row = row
        self.timestamp = timestamp
        self.symbol = symbol

    def __getitem__(self, field: str) -> float:
        try:
            col = self._columns[field]
        except KeyError:
            raise KeyError(f"{self.symbol}: no field {field!r} in bar") from None
        return float(self._values[self._row, col])

    def __contains__(self, field: object) -> bool:
        return field in self._columns

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def get(self, field: str, default: float | None = None) -> float | None:
        col = self._columns.get(field)
        if col is None:
            return default
        return float(self._values[self._row, col])

    def keys(self) -> tuple[str, ...]:
        return tuple(self._columns)

    @property
    def index(self) -> pd.Index:
        """Field names, for ``pd.Series``-compatible ``field in bar.index``."""
        return pd.Index(list(self._columns))

    def to_series(self) -> pd.Series:
        return pd.Series(
            self._values[self._row], index=list(self._columns), name=self.timestamp
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={self[k]:g}" for k in self._columns)
        return f"Bar({self.symbol} @ {self.timestamp}: {fields})"


class DataHandler(ABC):
    """Interface for anything that feeds bars into the engine."""

    symbols: list[str]

    @abstractmethod
    def update_bars(self) -> MarketEvent | None:
        """Advance one timestamp. Return a MarketEvent, or None when exhausted."""

    @abstractmethod
    def latest_bar(self, symbol: str) -> Bar | None:
        """Most recent bar visible for ``symbol`` (never a future bar)."""

    @abstractmethod
    def latest_price(self, symbol: str, field: str = "close") -> float | None:
        """Most recent value of ``field`` for ``symbol``."""


class HistoricCSVDataHandler(DataHandler):
    """Drive a backtest from in-memory OHLCV frames.

    Each frame must be indexed by a sorted, unique ``DatetimeIndex`` and contain
    at least a numeric ``close`` column. The handler unions all symbols'
    timestamps so a sparse symbol simply carries its last-known bar forward.

    Non-numeric columns are dropped: the hot path stores each frame as a single
    ``float64`` matrix, and a strategy that needs string metadata should not be
    reaching into the bar for it anyway.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not frames:
            raise ValueError("frames must contain at least one symbol")
        self.symbols = list(frames.keys())
        validated = {s: self._validate(s, df) for s, df in frames.items()}

        index: pd.Index | None = None
        for df in validated.values():
            index = df.index if index is None else index.union(df.index)
        assert index is not None  # guaranteed: frames is non-empty
        timeline = pd.DatetimeIndex(index)
        self._timeline: list[datetime] = list(timeline)
        self._n = len(self._timeline)

        # One vectorized alignment per symbol instead of one lookup per bar.
        # ``ffill`` maps each timeline slot to the last row at or before it, and
        # -1 for slots that precede the symbol's first bar — the same
        # no-look-ahead semantics the per-bar version had, computed up front.
        self._values: dict[str, np.ndarray] = {}
        self._columns: dict[str, dict[str, int]] = {}
        self._rows: dict[str, np.ndarray] = {}
        self._close_col: dict[str, int] = {}
        for symbol, df in validated.items():
            self._values[symbol] = np.ascontiguousarray(df.to_numpy(dtype=float))
            self._columns[symbol] = {c: i for i, c in enumerate(df.columns)}
            self._rows[symbol] = df.index.get_indexer(timeline, method="ffill")
            self._close_col[symbol] = self._columns[symbol]["close"]

        self._cursor = -1

    @staticmethod
    def _validate(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError(f"{symbol}: frame must have a 'close' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{symbol}: index must be a DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if not df.index.is_unique:
            raise ValueError(f"{symbol}: index must not contain duplicate timestamps")
        numeric = df.select_dtypes(include="number")
        if "close" not in numeric.columns:
            raise ValueError(f"{symbol}: 'close' column must be numeric")
        return numeric

    def update_bars(self) -> MarketEvent | None:
        cursor = self._cursor + 1
        if cursor >= self._n:
            return None
        self._cursor = cursor
        return MarketEvent(timestamp=self._timeline[cursor])

    def latest_bar(self, symbol: str) -> Bar | None:
        if self._cursor < 0:
            return None
        row = int(self._rows[symbol][self._cursor])
        if row < 0:
            return None  # symbol has no bar at or before "now"
        return Bar(
            self._values[symbol],
            self._columns[symbol],
            row,
            self._timeline[self._cursor],
            symbol,
        )

    def latest_price(self, symbol: str, field: str = "close") -> float | None:
        cursor = self._cursor
        if cursor < 0:
            return None
        row = int(self._rows[symbol][cursor])
        if row < 0:
            return None
        if field == "close":
            col = self._close_col[symbol]
        else:
            col = self._columns[symbol].get(field, -1)
            if col < 0:
                raise KeyError(f"{symbol}: no field {field!r} in frame")
        return float(self._values[symbol][row, col])
