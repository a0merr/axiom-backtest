"""Execution handlers: turn orders into fills with realistic frictions.

Two costs are modeled explicitly and separately:

* **Slippage** — the gap between the decision price and the actual fill. Modeled
  here as a fraction of price scaled by direction (you buy worse, sell worse).
  A volume/volatility-aware model can subclass ``SlippageModel``.
* **Commission** — explicit transaction cost, per-share plus a floor, the shape
  most retail/prime broker schedules take.

Ignoring these is the most common way a backtest lies: a strategy that trades
often can look brilliant gross and be ruinous net.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .data import DataHandler
from .event import Direction, FillEvent, OrderEvent


class SlippageModel(ABC):
    @abstractmethod
    def fill_price(self, reference_price: float, direction: Direction) -> float:
        ...


class PercentageSlippage(SlippageModel):
    """Fill moves against you by ``rate`` of the reference price."""

    def __init__(self, rate: float = 0.0005):
        if rate < 0:
            raise ValueError("slippage rate must be non-negative")
        self.rate = rate

    def fill_price(self, reference_price: float, direction: Direction) -> float:
        if direction == Direction.LONG:
            return reference_price * (1 + self.rate)
        if direction == Direction.SHORT:
            return reference_price * (1 - self.rate)
        return reference_price


class CommissionModel(ABC):
    @abstractmethod
    def commission(self, quantity: float, fill_price: float) -> float:
        ...


class PerShareCommission(CommissionModel):
    """Per-share cost with a per-order minimum (e.g. IBKR-style)."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        self.per_share = per_share
        self.minimum = minimum

    def commission(self, quantity: float, fill_price: float) -> float:
        return max(self.minimum, abs(quantity) * self.per_share)


class ExecutionHandler(ABC):
    """Turn orders into fills.

    Two timing hooks let the engine support both same-bar and next-bar fills
    without knowing which model is in use:

    * ``execute_order`` is called when an ORDER event is processed.
    * ``on_bar`` is called at the start of every bar, before strategy logic, so
      a deferred handler can fill orders queued on the previous bar using the
      new bar's price.
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ):
        self.data = data
        self.slippage = slippage or PercentageSlippage()
        self.commission_model = commission or PerShareCommission()

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ...

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        return []

    def _build_fill(
        self, order: OrderEvent, reference_price: float
    ) -> FillEvent:
        fill_price = self.slippage.fill_price(reference_price, order.direction)
        slippage_cost = abs(fill_price - reference_price) * order.quantity
        commission = self.commission_model.commission(order.quantity, fill_price)
        return FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )

    def _reference_price(self, symbol: str, field: str) -> float | None:
        """Latest value of ``field``, falling back to close if absent."""
        bar = self.data.latest_bar(symbol)
        if bar is None:
            return None
        if field in bar.index:
            return float(bar[field])
        return self.data.latest_price(symbol)  # close fallback


class SimulatedExecutionHandler(ExecutionHandler):
    """Fill at the current bar's close, plus frictions.

    Simple and conservative-ish once slippage is applied, but optimistic: a
    signal computed from a bar's close cannot in reality be traded at that same
    close. Use ``NextBarExecutionHandler`` to remove that look-ahead.
    """

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        ref = self.data.latest_price(order.symbol)
        if ref is None:
            return None
        return self._build_fill(order, ref)


class NextBarExecutionHandler(ExecutionHandler):
    """Fill at the *next* bar's open — the realistic, no-look-ahead model.

    Orders are queued when received and filled on the following bar using its
    open price (falling back to close if the data has no ``open`` column). The
    consequence is honest: a signal on bar *t* cannot be acted on until bar
    *t+1*, and an order generated on the final bar never fills (there is no next
    bar to trade into).
    """

    def __init__(
        self,
        data: DataHandler,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        price_field: str = "open",
    ):
        super().__init__(data, slippage, commission)
        self.price_field = price_field
        self._pending: list[OrderEvent] = []

    def execute_order(self, order: OrderEvent) -> FillEvent | None:
        self._pending.append(order)
        return None  # filled later, in on_bar

    def on_bar(self, event) -> list[FillEvent]:  # noqa: ANN001 - MarketEvent
        if not self._pending:
            return []
        fills: list[FillEvent] = []
        for order in self._pending:
            ref = self._reference_price(order.symbol, self.price_field)
            if ref is not None:
                fills.append(self._build_fill(order, ref))
        self._pending.clear()
        return fills
