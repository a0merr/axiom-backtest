"""The event loop that ties the components together.

Order of operations per bar matters and is fixed here:

1. ``update_bars`` advances the clock and exposes the new bar.
2. The strategy reacts to the bar and emits signals.
3. The portfolio sizes signals into orders.
4. The execution handler fills orders (applying costs/slippage).
5. The portfolio books the fills.
6. The portfolio marks equity to market *after* fills.

This sequencing means a signal on bar *t* is filled using bar *t*'s price and
reflected in bar *t*'s equity mark — costs included, no future leakage.
"""

from __future__ import annotations

from collections import deque

from .data import DataHandler
from .event import Event, EventType, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import ExecutionHandler
from .portfolio import Portfolio
from .strategy import Strategy


class BacktestEngine:
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: ExecutionHandler,
    ):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self._events: deque[Event] = deque()

    def run(self) -> Portfolio:
        while True:
            market = self.data.update_bars()
            if market is None:
                break
            # Fill orders queued on the previous bar at this bar's price before
            # the strategy reacts (no-look-ahead for deferred execution).
            for fill in self.execution.on_bar(market):
                self.portfolio.on_fill(fill)
            self._events.append(market)
            self._drain()
            # Mark to market once the bar's events are fully processed.
            self.portfolio.on_market(market)
        return self.portfolio

    def _drain(self) -> None:
        while self._events:
            event = self._events.popleft()
            if event.type == EventType.MARKET:
                assert isinstance(event, MarketEvent)
                for signal in self.strategy.on_market(event):
                    self._events.append(signal)
            elif event.type == EventType.SIGNAL:
                assert isinstance(event, SignalEvent)
                for order in self.portfolio.on_signal(event):
                    self._events.append(order)
            elif event.type == EventType.ORDER:
                assert isinstance(event, OrderEvent)
                fill = self.execution.execute_order(event)
                if fill is not None:
                    self._events.append(fill)
            elif event.type == EventType.FILL:
                assert isinstance(event, FillEvent)
                self.portfolio.on_fill(event)
