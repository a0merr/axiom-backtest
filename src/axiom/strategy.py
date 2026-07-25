"""Strategy interface and a reference implementation.

A strategy consumes ``MarketEvent``s and emits ``SignalEvent``s. It must never
reach into the future: it may only use ``data.latest_*`` accessors, which are
guaranteed to expose bars up to the current timestamp only.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from .data import DataHandler
from .event import Direction, MarketEvent, SignalEvent


class Strategy(ABC):
    @abstractmethod
    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        """Return zero or more signals for this bar."""


class MovingAverageCrossover(Strategy):
    """Classic long/flat trend filter.

    Go long when the fast SMA crosses above the slow SMA; exit when it crosses
    back below. Intentionally simple — its job in this repo is to exercise the
    engine end-to-end and serve as the honest "where it breaks" case study, not
    to make money.

    The averages are maintained incrementally: each bar adds the new price and
    subtracts the one leaving each window, so the cost per bar is constant
    rather than proportional to ``slow``. Recomputing both sums from scratch
    every bar is the standard way an indicator quietly turns an O(bars) backtest
    into an O(bars x window) one — invisible on a year of daily data, fatal on
    a decade of minutes.

    Running sums accumulate floating-point drift, so both are recomputed from
    the live window every ``_RESYNC_EVERY`` bars. That keeps the error bounded
    at a cost of O(slow) once every few thousand bars instead of every bar.
    """

    _RESYNC_EVERY = 4096

    def __init__(self, data: DataHandler, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.data = data
        self.fast = fast
        self.slow = slow
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=slow))
        self._fast_sum: dict[str, float] = defaultdict(float)
        self._slow_sum: dict[str, float] = defaultdict(float)
        self._invested: dict[str, bool] = defaultdict(bool)
        self._bars_seen = 0

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        self._bars_seen += 1
        resync = self._bars_seen % self._RESYNC_EVERY == 0
        signals: list[SignalEvent] = []
        for symbol in self.data.symbols:
            price = self.data.latest_price(symbol)
            if price is None:
                continue
            window = self._prices[symbol]
            n = len(window)
            # The deque is bounded at `slow`, so appending to a full one evicts
            # window[0]; that is exactly the term leaving the slow sum. The fast
            # sum sheds window[-fast] instead, once it is full.
            self._slow_sum[symbol] += price - (window[0] if n == self.slow else 0.0)
            self._fast_sum[symbol] += price - (
                window[-self.fast] if n >= self.fast else 0.0
            )
            window.append(price)
            if len(window) < self.slow:
                continue

            if resync:
                prices = list(window)
                self._slow_sum[symbol] = math.fsum(prices)
                self._fast_sum[symbol] = math.fsum(prices[-self.fast :])

            fast_ma = self._fast_sum[symbol] / self.fast
            slow_ma = self._slow_sum[symbol] / self.slow

            if fast_ma > slow_ma and not self._invested[symbol]:
                self._invested[symbol] = True
                signals.append(
                    SignalEvent(symbol, event.timestamp, Direction.LONG)
                )
            elif fast_ma < slow_ma and self._invested[symbol]:
                self._invested[symbol] = False
                signals.append(
                    SignalEvent(symbol, event.timestamp, Direction.EXIT)
                )
        return signals
