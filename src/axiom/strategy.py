"""Strategy interface and a reference implementation.

A strategy consumes ``MarketEvent``s and emits ``SignalEvent``s. It must never
reach into the future: it may only use ``data.latest_*`` accessors, which are
guaranteed to expose bars up to the current timestamp only.
"""

from __future__ import annotations

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
    """

    def __init__(self, data: DataHandler, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.data = data
        self.fast = fast
        self.slow = slow
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=slow))
        self._invested: dict[str, bool] = defaultdict(bool)

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        for symbol in self.data.symbols:
            price = self.data.latest_price(symbol)
            if price is None:
                continue
            window = self._prices[symbol]
            window.append(price)
            if len(window) < self.slow:
                continue

            prices = list(window)
            fast_ma = sum(prices[-self.fast :]) / self.fast
            slow_ma = sum(prices) / self.slow

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
